# Customer-R1 — Server Setup & Training

Server-side procedure for running SFT / GRPO on the H100 8-GPU cluster.
Data (Phase 1~3) was generated locally and committed under `data/processed*/`,
so a fresh clone is sufficient to start training.

This document captures the full install sequence we found necessary —
`pip install -r requirements.txt` alone is **not** sufficient because the
torch / flash-attn / NCCL / verl stack has version constraints pip cannot
resolve on its own.

## 1. Recommended OS / Python

| Choice | Recommended | Why |
|---|---|---|
| OS | **Ubuntu 22.04 LTS** | GLIBC 2.35, CUDA 12.x wheels work cleanly. Ubuntu 20 GLIBC 2.31 collides with new wheels |
| Python | **3.10** | verl/vllm/flash-attn ship cp310 wheels; cp312 wheels often lag |
| CUDA driver | **>= 535.x** (CUDA 12.2 runtime support) | torch 2.4+cu121 needs CUDA <= 12.2 driver |

## 2. Clone + venv

```bash
git clone https://github.com/km1216lim/customer-r1.git
cd customer-r1
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## 3. Install dependencies (ORDER MATTERS)

`requirements.txt` lists most packages, but **three** install steps are
special — `vllm` first (it pins torch), then `flash-attn` from a wheel URL
(building from source is fragile), then `verl --no-deps` (to avoid torch
downgrade). Follow the steps in order:

```bash
# Step 1. requirements.txt minus the special-case packages
pip install -r requirements.txt

# Step 2. vllm — pins torch 2.4.0+cu121 transitively
pip install vllm==0.6.3

# Step 3. Confirm torch + abi
python -c "import torch; print(torch.__version__, torch.version.cuda,
  torch._C._GLIBCXX_USE_CXX11_ABI)"
# Expect: 2.4.0+cu121 12.1 False

# Step 4. flash-attn from pre-built wheel
# (torch 2.4 + cu12.x + cp310 + cxx11abiFALSE)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Step 5. verl 0.4.1 — DO NOT let it touch torch
pip install verl==0.4.1 --no-deps

# Step 6. Pin transformers (verl/vllm both compatible; 4.46.3 is the sweet spot)
pip install "transformers==4.46.3"

# Step 7. NCCL: remove the cu13 leftover that vllm pulls in transitively
pip uninstall -y nvidia-nccl-cu13
pip install --force-reinstall --no-deps nvidia-nccl-cu12==2.20.5

# Step 8. pyarrow: verl 0.4.1 needs >=19
pip install "pyarrow>=19.0.0,<20.0.0"
```

## 4. Verification

Run all four checks before launching a long job:

```bash
# (a) Driver + GPU count
nvidia-smi
nvidia-smi topo -m              # 8x H100 expected, NVLink between

# (b) Torch / CUDA wire-up
python -c "import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('gpus', torch.cuda.is_available(), torch.cuda.device_count())"
# Expect: torch 2.4.0+cu121, cuda 12.1, gpus True 8

# (c) NCCL pin
python -c "import torch; print('nccl', torch.cuda.nccl.version())"
# Expect: nccl (2, 20, 5)

# (d) verl + flash-attn chain (the actual import path SFT goes through)
python -c "
from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
from verl.utils.config import OmegaConf
from flash_attn.bert_padding import pad_input
from transformers import AutoModel
print('verl SFT trainer OK')
"
# Expect: 'verl SFT trainer OK' with no ImportError.

# (e) Datasets are present
ls data/processed/         # baseline
ls data/processed_L1/      # L1 ablation
ls data/processed_L2/      # main compressed variant
ls data/processed_L1L2/    # L1+L2 ablation
```

If any check fails, see §7 trouble-shooting.

## 5. SFT training

```bash
# Baseline (paper reproduction)
bash scripts/launch.sh --gpus 8 --model 7b --stage sft
#   -> configs/sft_base.yaml -> data/processed/ -> ckpt/sft/

# L2 compressed (main hypothesis)
bash scripts/launch.sh --gpus 8 --model 7b --stage sft --data l2
#   -> configs/sft_l2.yaml -> data/processed_L2/ -> ckpt/sft-l2/
```

Notes on the launcher:
- Topology is inferred from `--gpus`/`--model`. `8_7b` is the main entry.
- `--data` selects the yaml: `baseline` (default) -> `sft_base.yaml`,
  `l2` -> `sft_l2.yaml`.
- NCCL env vars are container-friendly defaults (IB disabled, P2P disabled,
  blocking NCCL communicator). Override with `export VAR=...` before the
  launcher if running on a bare-metal NVLink+IB box.
- Paper §4.3: lr 1e-5, AdamW, warmup 150 step, cosine, 2000 step × bs 64.

### Resume from interruption

SFT saves every `save_every_n_steps` (200 by default; see
`configs/sft_base.yaml`). Add `--resume_from ckpt/sft-l2/global_step_1200`
to keep going from a specific checkpoint. **Do not change `--gpus` across
resume** — optimizer state is partitioned per GPU count.

### Different GPU count for SFT (smoke testing)

The four-stage compression hypothesis can be validated with 4 GPUs for
speed-of-iteration:
```bash
bash scripts/launch.sh --gpus 4 --model 3b --stage sft --data l2   # 3B smoke
bash scripts/launch.sh --gpus 4 --model 7b --stage sft --data l2   # 7B, tighter mem
```
`topology.yaml` declares `4_3b` and `4_7b` (experimental, ~70GB peak).

## 6. GRPO training (after SFT)

```bash
bash scripts/launch.sh --gpus 8 --model 7b --stage grpo            # baseline
bash scripts/launch.sh --gpus 8 --model 7b --stage grpo --data l2  # L2
```

GRPO loads vLLM in collocated mode by default (`8_7b` topology), training
and rollout time-slice on the same 8 GPUs. Difficulty-aware reward weights
(paper §3.2) are already in `configs/grpo_*.yaml`.

The actor and reference are both initialized from the matching SFT
checkpoint (`ckpt/sft/latest_hf` or `ckpt/sft-l2/latest_hf`).

## 7. Trouble-shooting (everything we hit)

| Symptom | Root cause | Fix |
|---|---|---|
| `Driver too old (found 12020)` warning + crashes | torch built for CUDA 13.0 was installed (e.g. torch 2.11+cu130) | Reinstall: `pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall` |
| `flash_attn_2_cuda.so: undefined symbol _ZN3c10...` | flash-attn ABI doesn't match torch | Reinstall the wheel matching torch 2.4 + cu12 + cp310 + abiFALSE (see §3 step 4) |
| `cannot import name 'OmegaConfig' from 'verl.utils.config'` | `train/sft.py` placeholder used a name verl 0.4.1 doesn't expose | Already fixed in `train/sft.py`: loads `verl/trainer/config/sft_trainer.yaml` and calls `run_sft(cfg)` |
| `verl 0.4.1 requires pyarrow>=19.0.0` | requirements pinned pyarrow<18 for local Windows | `pip install "pyarrow>=19.0.0,<20.0.0"` |
| `could not import module 'AutoModel'` | transformers 5.x installed (incompatible with verl 0.4.1) | `pip install "transformers==4.46.3"` |
| `NCCL WARN Cuda failure 'CUDA driver version is insufficient'` | cu13 NCCL leftover (`nvidia-nccl-cu13 2.28.9`) being loaded | `pip uninstall -y nvidia-nccl-cu13 && pip install --force-reinstall --no-deps nvidia-nccl-cu12==2.20.5` |
| `libnccl.so.2: cannot open shared object file` after removing nccl-cu13 | cu12/cu13 packages share the same `.so` path; uninstalling cu13 deletes it | The reinstall in the line above repairs `libnccl.so.2` |
| `NCCL timeout in communicator initialization` (after model load) | `TORCH_NCCL_USE_COMM_NONBLOCKING=1` (experimental, hangs on K8s pods) | Default in `scripts/launch.sh` is now 0; if overridden, re-export it as 0 |
| `cannot import name 'clip_grads_with_norm_' from 'torch.nn.utils.clip_grad'` | verl 0.4.1 calls a torch 2.5+ API; we pin torch 2.4 for vllm compat | Already fixed: `train/sft.py` shim + direct replacement of `verl.utils.fsdp_utils.fsdp2_clip_grad_norm_` with torch 2.4's `clip_grad_norm_` |
| `[error] base config not found: configs/sft_base.yaml` | running launcher from outside the repo root | `cd customer-r1` first; configs are referenced by relative path |
| Training stuck at `Epoch 1/4: 0%` for 5~15 min | First-step warmup (CUDA graphs, kernel compile, first FSDP forward) | Normal. Wait. nvidia-smi GPU-Util should be >0. Past 30 min with GPU-Util=0 means a real hang |
| Disk filling up | each checkpoint ~14 GB; 10 saved per 2000-step run | Add `train.max_ckpt_to_keep: 3` to `configs/sft_*.yaml` (keeps latest 3) |

## 8. Evaluation (Phase 6)

After SFT (and later GRPO) finishes, evaluate with the bundled launchers —
see `docs/evaluation.md` for the full guide.

```bash
# baseline + L2 in one shot
bash scripts/eval_all.sh --stage sft               # 8 GPUs, both variants
bash scripts/eval_all.sh --stage sft --tp_size 4   # 4 GPUs (recommended for Qwen num_kv_heads=4)
bash scripts/eval_all.sh --stage grpo --tp_size 4  # after Phase 5

# Specific variant / specific step / score-only
bash scripts/eval.sh --stage sft --data l2 --tp_size 4
bash scripts/eval.sh --stage sft --data l2 --step 200             # eval mid-run ckpt
bash scripts/eval.sh --stage sft --data l2 --skip_inference       # rescore existing JSONL
```

Eval is GPU-count independent — results are identical for `--tp_size 4`
and `--tp_size 8`, only wall time differs.

## 9. Time budget (single H100 8x, observed)

| Stage | Per variant | 2 variants total |
|---|---|---|
| SFT (2000 step × bs 64, 65K ctx) | ~30~40 h | ~60~80 h |
| GRPO (2 epoch × bs 64, rollout collocated) | ~24~36 h | ~48~72 h |
| Evaluation (vLLM inference + scoring) | ~10~20 min | ~30 min |
| **Total** | | **~5 days** |

Plan around the cluster's job-time limits. Checkpoint resume (`--resume_from
ckpt/.../global_step_N`) is supported but only with the same `--gpus`.

## 10. What gets committed vs. ignored

Tracked (`git clone` is enough to start training):
- `data/processed*/` parquets (~100 MB total)
- code, configs, prompts, docs, tests, scripts

Ignored (`.gitignore`):
- raw / synthesis JSONL: `data/trajectories/`, `data/trajectories_synth/`
- checkpoints: `ckpt/`, `wandb/`, `runs/`
- smoke / debug outputs: `data/_smoke_*`, `data/redundancy_*`
- secrets: `.env`, GCP service-account keys
- IDE: `.claude/`, `.vscode/`, etc.
