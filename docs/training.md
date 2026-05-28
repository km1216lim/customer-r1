# Customer-R1 — Server Setup & Training

Server-side checklist for running SFT / GRPO on the H100 8-GPU cluster.
Data (Phase 1~3) has already been generated locally and is committed to git
under `data/processed*/`, so a fresh clone is sufficient to start training.

## 1. Clone + virtualenv

```bash
git clone https://github.com/km1216lim/customer-r1.git
cd customer-r1

# Pick whichever Python venv your cluster uses.
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` covers torch / transformers / verl / vllm / flash-attn /
deepspeed / ray / pandas / pyarrow / wandb. Some packages (especially
`flash-attn`) need the right CUDA / torch ABI — if pip wheels fail, build
from source or pin to a verl-compatible release matching your CUDA.

## 2. Sanity checks before launching a long job

```bash
# (a) GPU visibility and topology
nvidia-smi
nvidia-smi topo -m            # 8× H100 expected, NVLink between them

# (b) Verify the data shards are present
ls data/processed/            # baseline:   train.parquet + test.parquet + manifest.json
ls data/processed_L2/         # main compressed variant
ls data/processed_L1/         # ablation
ls data/processed_L1L2/       # ablation
cat data/processed/manifest.json | head -30
cat data/processed_L2/manifest.json | head -30

# (c) Verify wandb auth (optional but recommended)
wandb login          # paste WANDB_API_KEY
```

If wandb is unavailable on the server, set `logging.use_wandb: false` in the
relevant `configs/*.yaml` and the trainer falls back to a console logger.

## 3. SFT training

The launcher infers topology from `--gpus` and `--model`, and selects the
data variant via `--data`.

```bash
# Baseline (paper reproduction)
bash scripts/launch.sh --gpus 8 --model 7b --stage sft
#   → uses configs/sft_base.yaml → data/processed/

# L2 compressed (main hypothesis)
bash scripts/launch.sh --gpus 8 --model 7b --stage sft --data l2
#   → uses configs/sft_l2.yaml → data/processed_L2/
```

Checkpoints land under `ckpt/sft/` (baseline) or `ckpt/sft-l2/` (L2). The
`run_name_prefix` field in each yaml controls the wandb run name and the
local directory suffix.

Paper hyperparameters (already encoded in `configs/sft_base.yaml`,
`configs/sft_l2.yaml`): lr 1e-5, AdamW, warmup 150 steps, cosine schedule,
2000 total steps × batch 64. Expected wall time on H100 8× single node:
~15–20 hours per variant.

## 4. GRPO training (after SFT)

```bash
# Baseline
bash scripts/launch.sh --gpus 8 --model 7b --stage grpo
#   → uses configs/grpo_base.yaml; actor/ref init from ckpt/sft/latest_hf

# L2
bash scripts/launch.sh --gpus 8 --model 7b --stage grpo --data l2
#   → uses configs/grpo_l2.yaml; actor/ref init from ckpt/sft-l2/latest_hf
```

GRPO loads vLLM in the collocated mode (default for 8_7b topology), so the
same 8 GPUs handle both training and rollout via verl's hybrid engine.

Difficulty-aware reward weights (paper §3.2) are already wired in
`configs/grpo_*.yaml` — `input`=2000, `hard_click`=1000,
`product_option`=10, `review/search/terminate`=1, `wrong_click_penalty`=-1.

## 5. Resume after interruption

SFT / GRPO save every `save_every_n_steps`. To resume from the latest
checkpoint, point the trainer at it explicitly:

```bash
bash scripts/launch.sh --gpus 8 --model 7b --stage sft --data l2 \
  --resume_from ckpt/sft-l2/step_1200
```

(Forwarded via the `EXTRA_ARGS` passthrough in the launcher.)

**Caveat — changing GPU count between resumes is unsafe.** Optimizer
state and dataloader random state are partitioned by the GPU count, so
resuming on a different `--gpus` value breaks the partition. If you must
switch (e.g. cluster is full at 8 GPU and only 4 available), prefer to
load weights only and restart the optimizer from scratch.

## 6. Evaluation

After GRPO completes for both variants:

```bash
python eval/next_action_acc.py \
  --ckpt ckpt/grpo/latest_hf       \
  --data data/processed/test.parquet
python eval/next_action_acc.py \
  --ckpt ckpt/grpo-l2/latest_hf    \
  --data data/processed_L2/test.parquet
```

Report shape: overall accuracy + per-click_type breakdown + (optional)
rationale BERTScore / ROUGE-L when `--rationale_metrics` is set.

## 7. Time budget summary (single H100 8×)

| Stage | Per-variant time | 2 variants total |
|---|---|---|
| SFT (2000 step × bs 64) | ~15–20 h | ~30–40 h |
| GRPO (2 epoch × bs 64) | ~24 h | ~48 h |
| Evaluation | ~30 min | ~1 h |
| **Total to finish baseline + L2** | | **~80–90 h (3.5–4 days)** |

Plan around the cluster's job-time limits; for shorter queues, use the
resume mechanism above.

## 8. What gets committed vs. ignored

Tracked (so a fresh `git clone` on the server is enough):
- code, configs, prompts, docs, tests
- `data/processed*/` parquet outputs of Phase 3 (≈100 MB total)

Not tracked (.gitignore):
- `data/trajectories/`, `data/trajectories_synth/` (697 MB + 843 MB raw / synth)
- `data/_smoke_*` (Phase 3 debug runs)
- `ckpt/`, `wandb/`, `runs/`
- `.env`, service-account keys
