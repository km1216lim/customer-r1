#!/usr/bin/env bash
# Install a venv-wide torch 2.4 -> 2.5 clip_grad backport.
#
# Why this exists:
#   verl 0.4.1's fsdp2_clip_grad_norm_ (verl/utils/fsdp_utils.py:482) does
#   `from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm`
#   at call time. Those private symbols only exist in torch 2.5+. We pin
#   torch 2.4.0 (for cu121 + vllm 0.6.3 + flash-attn 2.6.3 ABI), so the
#   import raises ImportError the first time the actor optimizer step runs.
#
# Why a Python-side patch in train/reward.py is not enough:
#   In verl's PPO topology, train/reward.py is imported only by the driver
#   process (TaskRunner — where reward scoring runs). The actor's optimizer
#   step runs in a separate Ray worker process that never imports our
#   reward module, so the driver-side monkey-patch never gets applied there.
#
# What this script does:
#   Writes a sitecustomize.py into the venv's site-packages. Python imports
#   sitecustomize automatically at interpreter startup, before any user code
#   or library runs. So every Python process started in this venv (driver,
#   Ray workers, vLLM workers, ...) gets the clip_grad backport applied
#   before verl can call fsdp2_clip_grad_norm_.
#
# Usage:
#   bash scripts/install_grpo_torch_compat.sh
#
# Idempotent — re-running just overwrites the same file.

set -euo pipefail

VENV_SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
TARGET="${VENV_SITE_PACKAGES}/sitecustomize.py"

cat > "${TARGET}" <<'PYEOF'
"""Customer-R1 venv-wide torch 2.4 -> 2.5 clip_grad backport.

Auto-imported by Python at interpreter startup. Adds the private clip_grad
helper symbols that verl 0.4.1's fsdp2_clip_grad_norm_ expects to import at
call time but that torch 2.4.0 doesn't ship — so every Python process in
this venv (driver, Ray workers, vLLM workers) sees the backport before any
verl code runs.

If torch 2.5+ is ever installed in this venv, the patch becomes a no-op
because `hasattr(_clip_grad, "_clip_grads_with_norm_")` will already be
True with the real implementation.
"""
try:
    import torch
    import torch.nn.utils.clip_grad as _clip_grad

    if not hasattr(_clip_grad, "_clip_grads_with_norm_"):
        def _get_total_norm(tensors, norm_type=2.0, error_if_nonfinite=False, foreach=None):
            if isinstance(tensors, torch.Tensor):
                tensors = [tensors]
            grads = [t for t in tensors if t is not None]
            if len(grads) == 0:
                return torch.tensor(0.0)
            device = grads[0].device
            norm_type = float(norm_type)
            if norm_type == float("inf"):
                norms = [g.detach().abs().max().to(device) for g in grads]
                total_norm = torch.max(torch.stack(norms))
            else:
                norms = [torch.linalg.vector_norm(g.detach(), norm_type).to(device) for g in grads]
                total_norm = torch.linalg.vector_norm(torch.stack(norms), norm_type)
            if error_if_nonfinite and not torch.isfinite(total_norm):
                raise RuntimeError(
                    f"Total norm of order {norm_type} for gradients is non-finite"
                )
            return total_norm

        def _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach=None):
            if isinstance(parameters, torch.Tensor):
                parameters = [parameters]
            grads = [p.grad for p in parameters if p is not None and p.grad is not None]
            if len(grads) == 0:
                return
            max_norm = float(max_norm)
            clip_coef = max_norm / (total_norm + 1e-6)
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            for g in grads:
                g.detach().mul_(clip_coef_clamped.to(g.device))

        # verl 0.4.1 uses both the private (underscore-prefixed) name and the
        # public name across different call sites — bind both to our impl.
        _clip_grad._get_total_norm = _get_total_norm
        _clip_grad._clip_grads_with_norm_ = _clip_grads_with_norm_
        _clip_grad.clip_grads_with_norm_ = _clip_grads_with_norm_
except Exception:
    # sitecustomize must never raise — Python startup would die.
    pass
PYEOF

echo "[install_grpo_torch_compat] wrote sitecustomize.py to:"
echo "  ${TARGET}"
echo ""
echo "Verification:"
python -c "from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm; print('  OK: backport active')"
echo ""
echo "Now (re)launch GRPO — every Python process started in this venv will"
echo "automatically apply the torch 2.4 -> 2.5 clip_grad backport at startup."
