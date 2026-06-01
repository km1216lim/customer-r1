# Customer-R1 — Evaluation Guide (Phase 6)

How to measure trained checkpoints against the Customer-R1 paper's Table 4
metrics, and how to compare `baseline` vs `L2` (and later `GRPO` variants).

## Pipeline overview

```
ckpt/sft-l2/global_step_2000/actor       (trained HF checkpoint)
        │
        ▼  eval/run_inference.py        (vLLM, tensor-parallel)
eval/preds_sft-l2.jsonl                  ({user_id, session_id, step_idx,
                                          completion, action_gt, rationale_gt,
                                          is_session_last_step})
        │
        ▼  eval/next_action_acc.py      (per-step + macro-F1 + session F1)
eval/preds_sft-l2.results.json + console table
```

Two stages: **infer** (slow, GPU) and **score** (fast, CPU only).

## Metrics produced (paper Table 4)

| Metric | What it measures | Counted on |
|---|---|---|
| **Next Action Gen.** | Full action match (type + name + text) | every step |
| **Action Type (Macro-F1)** | Per-class F1 averaged over click / input / terminate | every step |
| **Fine-grained Type** | Type match + required slots filled | every step |
| **Session Outcome F1** | Binary F1, "session ends in purchase" as the positive class | each session's last step |
| Format validity (diagnostic) | Share of completions that parse to a valid Action JSON | every step |
| Type accuracy (diagnostic) | Type-only correctness | every step |
| Per-user breakdown | All of the above, per `user_id` | grouped |

Optional (when `--rationale_metrics` is set):

| Metric | Notes |
|---|---|
| BERTScore F1 | needs `pip install bert-score` (in `requirements.txt`) |
| ROUGE-L F1 | needs `pip install rouge-score` (in `requirements.txt`) |

Only rows where both predicted and ground-truth rationales exist count for
rationale quality (most rows have only `rationale_synth`; paper rationale
metrics target the human-labeled subset).

---

## Step 1 — Inference (per variant)

Each compression variant **must be evaluated on the test parquet it was
trained with** — otherwise the prompt format (furniture markers, anchor
slicing) does not match the model's expectations.

```bash
# baseline SFT
python eval/run_inference.py \
  --model   ckpt/sft/global_step_2000/actor \
  --data    data/processed/test.parquet \
  --output  eval/preds_sft.jsonl \
  --tp_size 8

# L2 SFT
python eval/run_inference.py \
  --model   ckpt/sft-l2/global_step_2000/actor \
  --data    data/processed_L2/test.parquet \
  --output  eval/preds_sft-l2.jsonl \
  --tp_size 8
```

### Important flags

| Flag | Default | When to override |
|---|---|---|
| `--tp_size` | 8 | Match the GPU count you actually have at eval time |
| `--max_model_len` | 65536 | Must be ≥ the longest prompt_text + max_new_tokens. Matches training budget |
| `--max_new_tokens` | 512 | Paper completions are <200 tokens; 512 is a safe cap |
| `--temperature` | 0.0 | Greedy (deterministic). Use 0.7 only for sampling diagnostics |
| `--gpu_memory_utilization` | 0.9 | Drop to 0.7 if vLLM OOMs at weight load |
| `--enforce_eager` | off | Add for crash-isolation if CUDA graphs misbehave |

### Expected wall time (8x H100, vLLM)

Both test parquets are 992 samples. Greedy decode of completions averaging
~100 tokens → roughly **5~15 minutes per variant**. The output JSONL flushes
every batch, so you can watch progress in real time.

## Step 2 — Compute metrics

```bash
# baseline
python eval/next_action_acc.py \
  --predictions eval/preds_sft.jsonl \
  --out         eval/preds_sft.results.json \
  --rationale_metrics

# L2
python eval/next_action_acc.py \
  --predictions eval/preds_sft-l2.jsonl \
  --out         eval/preds_sft-l2.results.json \
  --rationale_metrics
```

The script prints Table 4 to the console (six decimal-aligned lines) and
writes the full breakdown — per-user, per-class type F1, session-outcome TP/FP
counts, rationale quality if enabled — to the `.results.json` file.

CPU only; runs in seconds. Drop `--rationale_metrics` if you want to skip
BERTScore/ROUGE-L (those add a few minutes the first time because BERTScore
downloads `roberta-large`).

## Step 3 — Compare baseline vs L2

The hypothesis: same 65K budget but L2 keeps ~10 more history steps per
sample → **same-or-better action accuracy**.

Side-by-side console view:

```bash
echo "=== baseline ===" && python -m json.tool eval/preds_sft.results.json     | head -25
echo "=== L2 ==="       && python -m json.tool eval/preds_sft-l2.results.json  | head -25
```

What to look at:

| Comparison | Reading |
|---|---|
| L2 `Next Action Gen.` ≥ baseline | ✅ hypothesis confirmed |
| L2 `Action Type Macro-F1` ≥ baseline | type discrimination preserved |
| `diagnostics.type_f1_per_class.click` | did L2 hurt the most common type? |
| `diagnostics.type_f1_per_class.input` | input form filling stayed robust? |
| `diagnostics.session_outcome.precision / recall` | purchase prediction balance |
| `per_user` spread | per-user consistency vs averaged improvement |

If L2 underperforms on a specific type (e.g. nav_bar clicks) but wins
overall, that's still a positive signal — it tells you which anchor-window
edges to revisit.

## Step 4 — GRPO checkpoints (Phase 5/6)

Identical commands; just point at the GRPO checkpoint dir.

```bash
python eval/run_inference.py \
  --model   ckpt/grpo-l2/global_step_400/actor   \
  --data    data/processed_L2/test.parquet       \
  --output  eval/preds_grpo-l2.jsonl             \
  --tp_size 8

python eval/next_action_acc.py \
  --predictions eval/preds_grpo-l2.jsonl              \
  --out         eval/preds_grpo-l2.results.json       \
  --rationale_metrics
```

Paper reference numbers (Qwen2.5-7B):

| Stage | Next Action Gen. |
|---|---|
| SFT only | ~0.49 |
| SFT + GRPO | ~0.55 |

Hitting close to these confirms paper reproduction; meaningful gain over
baseline at the same stage confirms the compression hypothesis.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `parse_model_output returns None` for many rows → `Next Action Gen. ≈ 0` | Model didn't learn the single-JSON output format yet | Train longer, check SFT loss curve, or set `--temperature 0` |
| vLLM OOM at weight load | `gpu_memory_utilization` too high | Drop to 0.7 |
| `max_model_len` exceeded errors | One test sample's prompt > model max | Lower `--max_new_tokens` or rerun tokenize_pack with stricter cap |
| BERTScore stalls | First-time download of `roberta-large` (~1.4 GB) | Wait once; it's cached afterwards |
| Session Outcome F1 == 0 | No predicted `click` with `click_type=purchase` | Check `diagnostics.session_outcome.tp/fp/fn` — model may not predict purchase at all |

## Where files end up

```
eval/
├── run_inference.py            # (this guide)
├── next_action_acc.py          # paper Table 4 scorer
├── preds_sft.jsonl             # baseline SFT predictions
├── preds_sft-l2.jsonl          # L2 SFT predictions
├── preds_grpo.jsonl            # baseline GRPO predictions  (Phase 5/6)
├── preds_grpo-l2.jsonl         # L2 GRPO predictions        (Phase 5/6)
├── preds_sft.results.json      # scorer output
├── preds_sft-l2.results.json
├── preds_grpo.results.json
└── preds_grpo-l2.results.json
```

`eval/` is tracked in git so the JSONL outputs become a reproducible record.
If you want them ignored (they can be tens of MB), add `eval/preds_*.jsonl`
to `.gitignore`.
