"""SFT entry point — verl 0.4.1 (FSDP2), topology-aware.

verl 0.4.1's `verl.trainer.fsdp_sft_trainer` exposes `run_sft(config)`; its
hydra entry is a thin wrapper:

    @hydra.main(config_path="config", config_name="sft_trainer", ...)
    def main(config): run_sft(config)

Rather than re-declaring verl's full config schema (which drifts across
versions), we load verl's *bundled* `config/sft_trainer.yaml` as the base
so all version-specific defaults are present, override only the fields
Customer-R1 needs, then call `run_sft(cfg)` directly under torchrun.

Data: data/processed*/{train,test}.parquet from tokenize_pack_compressed.py.
  prompt_key = "prompt_text"      (chat-templated, ends at the assistant
                                   generation prompt)
  response_key = "completion_text" (single-JSON paper format)
verl's single-turn SFTDataset tokenizes prompt + response and masks the
prompt span, computing loss on the response only.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

# --- torch 2.4 -> 2.5 API backport ----------------------------------------
# verl 0.4.1's fsdp2_clip_grad_norm_ imports
#   torch.nn.utils.clip_grad.{clip_grads_with_norm_, _get_total_norm}
# which only exist in torch 2.5+. We pin torch 2.4.0 for cu121 + vllm 0.6.3
# compatibility, so the two functions are missing. Backport them in-place
# BEFORE verl is imported, so verl's `from torch.nn.utils.clip_grad import ...`
# picks up our shims.
import torch
import torch.nn.utils.clip_grad as _clip_grad

if not hasattr(_clip_grad, "clip_grads_with_norm_"):
    def _get_total_norm(tensors, norm_type=2.0, error_if_nonfinite=False, foreach=None):  # noqa: D401
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
            raise RuntimeError("Total norm of order {} for gradients is non-finite".format(norm_type))
        return total_norm

    def clip_grads_with_norm_(parameters, max_norm, total_norm, foreach=None):  # noqa: D401
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

    _clip_grad._get_total_norm = _get_total_norm
    _clip_grad.clip_grads_with_norm_ = clip_grads_with_norm_


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", required=True)
    ap.add_argument("--topology_config", type=Path, default=Path("configs/topology.yaml"))
    ap.add_argument("--base_config", type=Path, default=Path("configs/sft_base.yaml"))
    ap.add_argument("--output_dir", type=Path, default=None)
    args = ap.parse_args()

    # Allow `python train/sft.py` and `torchrun train/sft.py` to import topology.py.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from topology import load_topology

    topo = load_topology(args.topology_config, args.topology)
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))

    from omegaconf import OmegaConf
    import verl.trainer.fsdp_sft_trainer as sft_mod
    from verl.trainer.fsdp_sft_trainer import run_sft

    # --- verl 0.4.1 fsdp2_clip_grad_norm_ replacement ---------------------
    # The torch 2.4 -> 2.5 shim above (which patches torch.nn.utils.clip_grad)
    # works when the shimmed import is exercised at lookup time, but verl's
    # fsdp_sft_trainer binds `fsdp2_clip_grad_norm_` AT MODULE LOAD via
    # `from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_`. The trainer
    # then calls the original verl function, whose internal
    # `from torch.nn.utils.clip_grad import clip_grads_with_norm_, ...` line
    # fires on every step regardless of our shim and crashes on torch 2.4.
    # Replace the verl function object directly in both modules with a
    # torch 2.4-native equivalent (torch.nn.utils.clip_grad.clip_grad_norm_,
    # which has the same effect — compute total norm and rescale grads).
    import torch.nn.utils.clip_grad as _clip_grad_mod
    import verl.utils.fsdp_utils as _verl_fsdp_utils

    def _customer_r1_fsdp2_clip_grad_norm_(parameters, max_norm, norm_type=2.0):
        return _clip_grad_mod.clip_grad_norm_(parameters, max_norm, norm_type)

    _verl_fsdp_utils.fsdp2_clip_grad_norm_ = _customer_r1_fsdp2_clip_grad_norm_
    sft_mod.fsdp2_clip_grad_norm_ = _customer_r1_fsdp2_clip_grad_norm_

    # Load verl's bundled base config so version-specific defaults stay intact.
    verl_yaml = Path(sft_mod.__file__).parent / "config" / "sft_trainer.yaml"
    cfg = OmegaConf.load(str(verl_yaml))
    # Allow overriding keys that may not exist in older bundled yamls.
    OmegaConf.set_struct(cfg, False)

    # --- data --------------------------------------------------------------
    cfg.data.train_files = base["data"]["train_path"]
    cfg.data.val_files = base["data"]["val_path"]
    cfg.data.prompt_key = "prompt_text"
    cfg.data.response_key = "completion_text"
    cfg.data.max_length = int(topo.context_length)
    cfg.data.train_batch_size = int(base["train"]["batch_size"])
    cfg.data.micro_batch_size_per_gpu = int(topo.per_device_micro_batch)

    # --- model -------------------------------------------------------------
    cfg.model.partial_pretrain = topo.model_name
    cfg.model.enable_gradient_checkpointing = True

    # --- top-level parallelism / padding ----------------------------------
    cfg.ulysses_sequence_parallel_size = int(topo.sp_size)
    cfg.use_remove_padding = True

    # --- optim (paper §4.3: lr 1e-5, warmup 150 steps, cosine) -------------
    total_steps = int(base["train"]["total_steps"])
    cfg.optim.lr = float(base["optim"]["lr"])
    cfg.optim.weight_decay = float(base["optim"]["weight_decay"])
    cfg.optim.lr_scheduler = base["optim"]["lr_scheduler"]
    cfg.optim.clip_grad = float(base["optim"]["max_grad_norm"])
    # verl uses a warmup *ratio* (fraction of total steps); the paper gives an
    # absolute step count, so convert.
    cfg.optim.warmup_steps_ratio = float(base["optim"]["warmup_steps"]) / total_steps

    # --- trainer -----------------------------------------------------------
    nnodes = int(os.environ.get("NNODES", "1"))
    cfg.trainer.total_training_steps = total_steps
    # verl 0.4.1 enforces BOTH total_training_steps and total_epochs and stops
    # at whichever is reached first. The bundled sft_trainer.yaml ships with
    # total_epochs=4, which on OPeRA-filtered (76 steps/epoch) gives 304
    # steps — well below paper's 2000. Override to a value high enough that
    # total_training_steps is the only effective cap.
    cfg.trainer.total_epochs = 9999
    cfg.trainer.project_name = base["logging"]["project"]
    cfg.trainer.experiment_name = f"{base['logging']['run_name_prefix']}-{topo.key}"
    out_dir = str(args.output_dir) if args.output_dir else f"ckpt/{base['logging']['run_name_prefix']}"
    cfg.trainer.default_local_dir = out_dir
    cfg.trainer.logger = ["console", "wandb"] if base["logging"].get("use_wandb") else ["console"]
    cfg.trainer.save_freq = int(base["train"]["save_every_n_steps"])
    cfg.trainer.test_freq = int(base["train"]["eval_every_n_steps"])
    cfg.trainer.seed = int(base["train"]["seed"])
    cfg.trainer.nnodes = nnodes
    cfg.trainer.n_gpus_per_node = int(topo.world_size) // nnodes

    if int(os.environ.get("RANK", "0")) == 0:
        print("[sft] resolved config:")
        print(OmegaConf.to_yaml(cfg))

    run_sft(cfg)


if __name__ == "__main__":
    main()
