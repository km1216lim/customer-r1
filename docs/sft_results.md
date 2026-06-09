# Customer-R1 SFT 단계 결과 — Baseline vs L2 압축

작성: 2026-06-09 · 상태: SFT 종료, GRPO 대기

paper(arxiv 2510.07230) Table 4의 4개 지표를 기준으로 baseline(uncompressed
prompt)과 L2(furniture dedup + action-anchored slicing 압축)를 비교한다.
GRPO 단계의 최종 결과는 본 문서가 아니라 [grpo_results.md](grpo_results.md)
(작성 예정)에 기록한다.

## 1. 학습 설정

- 모델: Qwen/Qwen2.5-7B-Instruct-1M
- Context: 65,536 tokens · effective batch: 64
- 학습 길이: 2,000 step (cosine LR, warmup 150, peak 1e-5)
- 하드웨어: 8× H100 (NVLink full-mesh 확인), `topology=8_7b`
- 학습 시간: 약 138시간 (≈ 5.8일), 평균 4.2 분/step
- 데이터: OPeRA-filtered (train 4,864 / test 992)
- 평가: `bash scripts/eval.sh --stage sft --data {baseline,l2}` — vLLM TP=8
  inference + Table 4 채점. Macro-F1은 `__INVALID__` pseudo-class를
  제외하고 canonical 3 클래스(`click`, `input`, `terminate`)에 대해서만
  평균 ([eval/next_action_acc.py](../eval/next_action_acc.py)).

## 2. Step 2000 최종 결과

| 지표 | Baseline | L2 (ours) | Δ (L2 − Baseline) |
|---|---|---|---|
| Next Action Gen. | 25.10 | 23.79 | −1.31 |
| Action Type (Macro-F1) | 44.39 | **46.24** | **+1.85** |
| Fine-grained Type | **80.24** | 71.88 | −8.36 |
| Session Outcome | 42.42 | **53.52** | **+11.10** |

(모든 값은 백분율)

### 2.1 Paper Table 4와의 비교

| 지표 | Baseline | L2 | Paper SFT-only | Baseline 격차 | L2 격차 |
|---|---|---|---|---|---|
| Next Action Gen. | 25.10 | 23.79 | 35.14 | −10.04 | −11.35 |
| Action Type (Macro-F1) | 44.39 | 46.24 | 72.66 | −28.27 | −26.42 |
| Fine-grained Type | 80.24 | 71.88 | 56.43 | **+23.81** | **+15.45** |
| Session Outcome | 42.42 | 53.52 | 66.29 | −23.87 | −12.77 |

- **FG-Type는 두 run 모두 paper를 큰 폭으로 추월** — 우리 모델의 format
  adherence(JSON 구조 + slot 채움)가 paper 수준 이상으로 학습됨.
- **Macro-F1은 두 run 모두 paper 대비 약 −26~28p** — `__INVALID__` 제외
  보정 후에도 여전한 격차. paper의 SFT 학습 schedule / 데이터 처리 / 또는
  metric 계산 정의에 우리와 다른 부분이 있을 가능성. GRPO 단계에서
  action-level reward로 직접 보정해야 좁혀질 영역.
- **Session Outcome은 L2가 paper에 가장 가까움** (−12.77p). 압축이 세션
  종료 판단을 무너뜨리지 않고 오히려 paper에 근접시킴.

### 2.2 Step 별 trajectory (5 ckpt × 2 run)

학습 진행 도중 저장된 중간 ckpt 결과:

| Step | Run | NAG | Macro-F1\* | FG-Type | Session |
|---|---|---|---|---|---|
| 200 | baseline | 3.12 | 25.82 | 82.56 | 52.78 |
| 200 | L2 | 1.81 | 25.51 | 82.86 | 46.38 |
| 600 | baseline | 16.94 | 29.07 | 72.98 | 23.73 |
| 600 | L2 | 16.83 | 29.34 | 72.98 | 23.73 |
| 1000 | baseline | 21.98 | 29.99 | 78.83 | 28.57 |
| 1000 | L2 | 20.97 | 30.58 | 79.33 | 38.81 |
| 1800 | baseline | 26.11 | 36.34 | 81.65 | 48.57 |
| 1800 | L2 | 23.89 | 35.38 | 68.35 | 37.50 |
| **2000** | **baseline** | 25.10 | **44.39** | 80.24 | 42.42 |
| **2000** | **L2** | 23.79 | **46.24** | 71.88 | **53.52** |

\* Macro-F1: step 2000만 INVALID 제외 보정값. step 200~1800은 보정 전 값
(약 +10~12p 더해야 보정값).

trajectory의 두 가지 특징:

1. **NAG는 단조 증가** — 학습이 길어질수록 exact match 정확도가 꾸준히 향상.
   step 1800이 두 run 모두 NAG 최선치 (baseline 26.11, L2 23.89), step 2000은
   미세하게 회귀하지만 본질적으로 같은 자릿수.
2. **FG-Type / Session은 비단조** — step 600 부근에 최저점이 있고 다시 회복.
   step 1800에서 baseline은 81.65 / 48.57로 회복했으나 L2는 68.35 / 37.50으로
   처짐 (terminate over-prediction phase). step 2000의 마지막 200 step
   cosine LR 감쇠에서 **L2가 Session에서 +16p 폭발적 회복** (37.50 → 53.52),
   반대로 baseline은 미세 악화 (48.57 → 42.42).

## 3. 압축 가설 평가

### 3.1 가설 ([compression_design.md](compression_design.md) §1)

> 같은 65K token budget 안에서 L2 압축이 paper baseline 대비 같거나 더 많은
> 의미 있는 history step을 담아 next-action 정확도를 같거나 향상시킬 수 있다.

### 3.2 결과 해석

step 2000 시점의 L2 vs baseline 직접 비교:

| 평가 축 | 결과 | 해석 |
|---|---|---|
| Next Action Gen | L2 −1.31p | 약한 손실. exact match 정확도는 압축으로 작게 떨어짐. |
| **Action Type (Macro-F1)** | **L2 +1.85p** | **양성.** 클래스 균형 잡힌 의사결정에서 L2가 우위. |
| Fine-grained Type | L2 −8.36p | 명확한 손실. format-level slot 채움 정확도에 압축 비용 있음. |
| **Session Outcome** | **L2 +11.10p** | **강한 양성.** 세션 종료/구매 판단에서 L2가 큰 우위. |

**핵심 메시지**: 압축은 **format-level 정밀도(FG-Type)에서 비용**을 지불하지만
**의사결정 품질(Macro-F1 + Session Outcome)에서는 baseline을 추월**한다.
중간 ckpt(step 600~1000)에서 두 run이 거의 동일했고, 마지막 cosine LR 감쇠
구간에서 L2가 Session에서 +16p 폭발적 회복을 보이며 baseline은 −6p 회귀한
것이 step 2000 격차의 직접 원인.

### 3.3 잠정 결론

1. **압축 가설 부분 입증**. L2가 Macro-F1과 Session Outcome에서 baseline을
   추월. 본 실험에서 paper와 가장 가까운 Session Outcome 값(53.52%)이
   L2에서 나옴 — paper 66.29와 격차 −13p로 baseline의 −24p 대비 절반.
2. **압축의 비용은 FG-Type 슬롯 정확도**. L2가 click의 semantic_id 같은
   세부 슬롯을 baseline 대비 8p 덜 정확하게 채움. 다만 NAG 격차는 −1p로
   작아 실용적 영향은 제한적.
3. **paper의 Macro-F1 72.66은 본 실험 두 run 모두 도달 못 함**. INVALID 제외
   보정만으로는 못 메우는 구조적 차이가 있으며, 이는 압축 효과와 별개로
   존재. GRPO가 action-level reward로 직접 다루는 영역.

## 4. 평가 코드 수정 사항

본 보고서의 Macro-F1은 다음 수정 후 계산:

- `eval/next_action_acc.py`의 `macro_f1` 함수가 이전엔 관찰된 모든
  label(`click`, `input`, `terminate`, `__INVALID__`)을 분모에 포함
  → INVALID는 GT support=0이고 phantom 4번째 class로 평균을 약 12p 끌어내림
- 수정: `average_labels` 파라미터 추가, default `("click", "input",
  "terminate")`. per-class 진단에는 INVALID가 그대로 표시되어
  diagnostics는 보존. INVALID 예측은 여전히 진짜 클래스의 false-negative로
  반영되어 페널티 유지.
- 결과: step 2000 baseline Macro-F1 33.30 → 44.39 (+11.09), L2 34.68 →
  46.24 (+11.56). 다른 3개 지표(NAG, FG-Type, Session)는 별도 함수라
  영향 없음.

## 5. GRPO 단계 진행 (대기)

paper의 SFT→GRPO 점프(Macro-F1 +X / Session +Y, paper Table 4 SFT+RL 행
참조)를 우리 환경에서도 재현 시도. GRPO 초기 ckpt 선택:

- baseline: `ckpt/sft/global_step_2000` (latest_hf 심링크)
- L2: `ckpt/sft-l2/global_step_2000` (latest_hf 심링크)

학습 설정 ([configs/grpo_{base,l2}.yaml](../configs/grpo_l2.yaml)):
- 2 epoch over OPeRA-filtered train (76 step/epoch × 2 ≈ 152 step)
- Difficulty-aware verifiable reward: input=2000, hard_click=1000,
  product_option=10, review/search/terminate=1, wrong_click=−1
- Topology: 8 GPU collocated (rollout time-sliced with training)
- 예상 시간: 약 30시간 per run

GRPO 종료 후 본 문서와 같은 형식으로
[grpo_results.md](grpo_results.md)에 baseline + L2 결과 정리. 압축 가설의
최종 평가는 GRPO 결과 비교를 통해 확정:

- **L2 GRPO ≥ baseline GRPO** → 압축이 SFT 우위를 RL에서도 유지. 가설 확정.
- **L2 GRPO ≈ baseline GRPO** → RL이 양 trajectory를 같은 attractor로
  끌어감. 압축은 비용 측면에서 의미 (효율 절감, 추론 throughput) 있지만
  품질 우위는 RL이 흡수.
- **L2 GRPO < baseline GRPO** → SFT 우위가 RL에서 사라짐. 압축의 의미가
  추가 ablation으로 좁아짐.

## 6. 데이터·실험 재현 정보

- 학습 명령:
  ```bash
  bash scripts/launch.sh --gpus 8 --model 7b --stage sft --data baseline
  bash scripts/launch.sh --gpus 8 --model 7b --stage sft --data l2
  ```
- 평가 명령:
  ```bash
  bash scripts/eval.sh --stage sft --data baseline
  bash scripts/eval.sh --stage sft --data l2
  ```
- 압축 효과 측정: [tokenize_pack_compressed.py](../data/tokenize_pack_compressed.py)
  의 `_stats_*` 출력 (per-step compression ratio p50/p90/p99 포함)
- 디자인 배경: [compression_design.md](compression_design.md)
- 학습 환경 셋업: [training.md](training.md)
- 평가 절차: [evaluation.md](evaluation.md)
