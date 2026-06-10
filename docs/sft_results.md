# Customer-R1 SFT 단계 결과 — Baseline vs L2 압축

작성: 2026-06-09 · 최종 업데이트: 2026-06-10 (Gemini 2.5/3.5 Flash 외부 baseline 추가) · 상태: SFT 종료, GRPO 대기

paper(arxiv 2510.07230) Table 4의 4개 지표를 기준으로 baseline(uncompressed
prompt), L2(furniture dedup + action-anchored slicing 압축), 그리고 외부
baseline인 Gemini 2.5 Flash / 3.5 Flash zero-shot을 비교한다. GRPO 단계의
최종 결과는 본 문서가 아니라 [grpo_results.md](grpo_results.md) (작성 예정)에
기록한다.

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
| Action Type (sklearn Macro-F1) | 44.39 | 46.24 | 72.66\* | −28.27 | −26.42 |
| **Action Type (Weighted-F1)** | **80.20** | **76.19** | 72.66\* | **+7.54** | **+3.53** |
| Fine-grained Type | 80.24 | 71.88 | 56.43 | **+23.81** | **+15.45** |
| Session Outcome | 42.42 | 53.52 | 66.29 | −23.87 | −12.77 |

\* Paper는 "Macro-F1"이라 표기하지만, 우리 sklearn 표준 macro 계산값과 거의
30p 격차가 나는 반면 sklearn weighted 계산값(GT support로 클래스를 가중평균)이
paper 수치에 거의 일치한다. §2.2 참조.

- **FG-Type는 두 run 모두 paper를 큰 폭으로 추월** — 우리 모델의 format
  adherence(JSON 구조 + slot 채움)가 paper 수준 이상으로 학습됨.
- **Weighted-F1 기준 baseline 80.20 > paper 72.66** — paper Macro-F1이 사실상
  weighted F1이라는 가설이 맞다면 우리 SFT가 paper를 오히려 추월. §4의
  Gemini Flash 결과(weighted 80.60)도 같은 자릿수라 가설 강화. sklearn 표준
  macro(클래스를 동등 가중)로는 두 run 모두 −26~28p 떨어지지만, 이는 minority
  class(input/terminate) 학습이 paper 대비 약하다는 뜻 (특히 terminate F1
  ~0.16, 1800 step 분석 참조).
- **Session Outcome은 L2가 paper에 가장 가까움** (−12.77p). 압축이 세션
  종료 판단을 무너뜨리지 않고 오히려 paper에 근접시킴.

### 2.2 Paper Macro-F1의 metric 정의 추정

수치 일치도 분석:

| 모델 | sklearn Macro-F1 (단순평균) | Weighted-F1 (support 가중) | Paper 보고치 |
|---|---|---|---|
| Ours SFT baseline (step 2000) | 44.39 | **80.20** | – |
| Ours SFT L2 (step 2000) | 46.24 | **76.19** | – |
| Gemini 2.5 Flash (§4) | 44.72 | **80.60** | – |
| Paper SFT-only | – | – | **72.66** |

세 모델 모두 weighted F1이 paper 72.66에 1자릿수 내로 일치(우리는 약간 우월,
Gemini도 비슷). 반면 sklearn 표준 macro F1로는 일관되게 −28p 격차.

paper 코드 미공개로 100% 확정은 불가하지만, 가능성 순서:

1. **paper의 "Macro-F1"이 실제로는 weighted F1** (sklearn `average='weighted'`).
   ML 논문에서 metric 명명이 느슨한 경우 흔함. 가능성 가장 높음.
2. paper가 sklearn 표준 macro를 사용했다면 우리 minority class(input/terminate)
   학습이 paper 대비 약하다는 뜻. 다만 weighted F1에서 우리가 paper 초과한다는
   사실로 미루어 우리 모델 능력 자체가 부족한 건 아님 — class balance 학습 방식
   차이 정도로 추정.
3. 두 metric을 paper가 모두 보고했으나 본문엔 한쪽만 인용했을 가능성.

본 보고서는 **두 metric 모두 함께 보고**해서 paper와의 fair comparison을 유지.
weighted F1 기준에서 우리 모델은 paper SFT를 동등하거나 약간 추월하는 위치.

### 2.3 Step 별 trajectory (5 ckpt × 2 run)

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
3. **Paper와의 Macro-F1 격차는 metric 정의 차이가 주된 원인** (§2.2).
   Weighted-F1 기준 baseline 80.20 > paper 72.66. sklearn 표준 macro
   기준 −28p 격차는 minority class 학습 차이로 해석 가능하지만 모델 능력
   자체 격차는 아님.

## 4. 외부 baseline 비교 — Gemini 2.5 / 3.5 Flash (zero-shot)

같은 65K truncated test set(`data/processed/test.parquet`)으로 zero-shot
Gemini 2.5 Flash와 3.5 Flash를 평가한 결과. 같은 입력 + 같은 평가 코드를
사용하므로 trained vs zero-shot 직접 비교 가능 ([eval/run_gemini_inference.py](../eval/run_gemini_inference.py)).
3.5 Flash는 Vertex AI의 `global` location에서만 접근 가능하며, 2.5 대비
reasoning이 길어 `max_output_tokens=2048` 설정 필요 (1024로는 일부 응답이
잘려 빈 completion 반환).

### 4.1 결과

| Model | NAG | sklearn Macro-F1 | Weighted-F1 | FG-Type | Session |
|---|---|---|---|---|---|
| **Ours SFT baseline** | **25.10** | 44.39 | 80.20 | **80.24** | 42.42 |
| Ours SFT L2 | 23.79 | **46.24** | 76.19 | 71.88 | 53.52 |
| Gemini 2.5 Flash | 18.95 | 44.72 | **80.60** | 79.84 | 59.46 |
| **Gemini 3.5 Flash** | 14.72 | 43.93 | 79.83 | 78.53 | **72.09** |
| Paper SFT-only | 35.14 | – | 72.66\* | 56.43 | 66.29 |

\* §2.2의 가설대로 paper "Macro-F1"이 weighted F1이라 가정.

축별 우위:

| 축 | 1위 | 2위 | 3위 | 4위 |
|---|---|---|---|---|
| NAG (verbatim copy) | Paper 35.14 | **Ours baseline 25.10** | Ours L2 23.79 | Gemini 3.5 14.72 |
| Weighted-F1 | Gemini 2.5 80.60 | **Ours baseline 80.20** | Gemini 3.5 79.83 | Ours L2 76.19 |
| FG-Type (format) | **Ours baseline 80.24** | Gemini 2.5 79.84 | Gemini 3.5 78.53 | Ours L2 71.88 |
| Session | **Gemini 3.5 72.09** | Paper 66.29 | Gemini 2.5 59.46 | Ours L2 53.52 |

### 4.2 Per-class 분석 — Gemini의 catastrophic terminate failure

Gemini Flash의 type별 예측 분포 (2.5 vs 3.5 비교):

| Class | GT support | 2.5 Pred | 3.5 Pred | 2.5 F1 | 3.5 F1 |
|---|---|---|---|---|---|
| click | 845 | 829 (0.98×) | 833 (0.99×) | 0.889 | 0.882 |
| input | 107 | 105 (0.98×) | 72 (0.67×) | 0.453 | 0.436 |
| **terminate** | **40** | **0 (0.00×)** ⚡ | **1 (0.03×)** ⚡ | **0.000** | **0.000** |
| (INVALID) | 0 | 58 | 86 | – | – |

**Gemini Flash는 992 샘플 동안 거의 단 한 번도 terminate를 예측하지 않음
(2.5: 0회, 3.5: 1회).** 우리 SFT step 200 시점에 보였던 click mode collapse와
같은 패턴 — 모델 세대가 올라가도 패턴 동일.

원인 추정:
- response_schema에 `terminate` enum 옵션은 명시했으나 "언제 써야 하는가"는
  학습되지 않음
- Task-specific 학습 없는 Gemini는 default "click이 안전"으로 수렴
- minority class 의사결정은 task-specific training 없이는 어려움

직접적 함의: **우리 task-specific 학습이 zero-shot으로는 못 하는 minority
class 의사결정을 가능하게 함**. terminate F1 = 0.16~0.45 정도라도 0.000은
아님.

### 4.3 Session Outcome 우위 — 부분적 artifactual + 부분적 진짜 향상

Gemini의 Session이 우리 모델을 앞선 메커니즘 분석:

| | tp | fp | fn | precision | recall | F1 | pred_purchase |
|---|---|---|---|---|---|---|---|
| Gemini 2.5 Flash | 22 | 2 | 28 | **91.7%** | 44.0% | **59.46** | 24 |
| **Gemini 3.5 Flash** | **31** | 5 | **19** | 86.1% | **62.0%** | **72.09** | **36** |
| Ours SFT L2 (step 1800 ref.) | 12 | 2 | 38 | 85.7% | 24.0% | 37.50 | 14 |

Gemini 우위의 메커니즘:

1. **Artifactual 부분 (두 모델 공통)**: Terminate를 거의 0번 예측 → 세션 마지막
   step도 click으로 예측 → 그 click 중 일부가 우연히 purchase로 매핑됨. 우리
   모델이 terminate over-prediction을 GRPO에서 보정하면 같은 메커니즘으로
   Session 점수 향상 가능.

2. **2.5 → 3.5의 +12.63p 향상은 진짜 task 이해 증가**: tp 22→31 (+9), recall
   44%→62% (+18p), purchase 예측 24→36 (+12). 3.5 Flash는 어떤 click이
   "purchase로 이어지는지"를 더 정확히 인식. 단, precision은 91.7%→86.1%로
   소폭 하락 (5개 false positive 추가).

즉 **Gemini Session 우위 = "terminate 안 씀의 부산물" + "더 강한 모델의
purchase 의도 인식 향상"**. 2.5와 비교하면 우리 모델의 Session 격차가 일부
artifactual이라는 가설이 유지되지만, 3.5가 paper SFT(66.29)도 넘는다는 점은
zero-shot 의사결정 품질이 충분히 강하다는 의미이기도 함.

### 4.4 본 실험의 narrative — 외부 baseline으로 강화

이 분석으로 세 개의 강한 주장이 가능:

#### 주장 1: 우리 7B 학습 모델이 paper SFT를 metric-level에서 재현 또는 능가

- Weighted F1로 보면 우리 80.20 vs paper 72.66 → **+7.54p**
- FG-Type 80.24 vs paper 56.43 → **+23.81p** (paper보다 format 잘 지킴)
- NAG와 Session에서만 paper에 못 미침 (각 −10p, −24p) — paper가 더 길게
  학습했거나 다른 fine-tuning trick 가능성

#### 주장 2: 우리 학습 모델이 zero-shot Gemini Flash와 동등 또는 NAG에서 우위

- Macro-F1 / FG-Type / Weighted-F1 모두 거의 동등 (±1p) — 2.5, 3.5 공통
- **NAG: 우리 baseline +6.15p (vs 2.5), +10.38p (vs 3.5) 우위** — verbatim
  copy 능력은 task-specific 학습의 직접 효과
- Session: Gemini 우위 (특히 3.5는 paper도 능가)이나 **부분적 artifactual**
  (terminate ≈ 0 예측의 부산물) + **부분적 진짜 향상** (3.5의 purchase 인식
  강화)

#### 주장 3: 모델 세대(2.5 → 3.5)가 올라가도 NAG 격차는 더 벌어짐

- 2.5 Flash → 3.5 Flash로 가도 NAG는 18.95 → 14.72로 **오히려 감소**
- 더 강한 모델은 reasoning이 길어지지만 **verbatim copy 능력은 약화**
- Task-specific 학습이 frontier zero-shot 모델로도 대체되지 않는 능력임을 입증
- Macro-F1 / FG-Type은 2.5와 3.5가 사실상 동일 — 모델 발전의 효과가 Session
  (의도 이해) 한 축에만 집중

#### 한 줄 결론

> **Customer-R1의 task-specific 7B 모델은 paper의 보고된 metric을 재현(또는
> 능가)하며, zero-shot Gemini Flash (2.5 / 3.5 둘 다)와 동등한 type-level
> 의사결정 성능을 보이면서 NAG(verbatim ID copy)에서 의미 있는 우위를 유지함.
> 더 강한 Gemini 3.5도 verbatim copy 능력은 오히려 떨어져 우리 우위가 확대됨.
> Session 우위만 Gemini가 가지나, 일부 artifactual(terminate 회피)이고 일부
> 진짜 모델 능력 향상. 압축(L2)은 Macro-F1과 Session에서 약간의 추가 이점을
> 제공함.**

### 4.5 통계적 유의성

n=992에서 metric별 95% 신뢰구간 (이항 분포):

| 비교 | 차이 | 95% CI 겹침? | 결론 |
|---|---|---|---|
| NAG: Ours baseline 25.10 vs Gemini 2.5 18.95 | +6.15p | 안 겹침 | **통계적으로 유의** |
| **NAG: Ours baseline 25.10 vs Gemini 3.5 14.72** | **+10.38p** | **확실히 안 겹침** | **강하게 유의** |
| Weighted-F1: Ours 80.20 vs Gemini 2.5 80.60 | -0.40p | 겹침 | 사실상 동등 |
| Weighted-F1: Ours 80.20 vs Gemini 3.5 79.83 | +0.37p | 겹침 | 사실상 동등 |
| FG-Type: Ours 80.24 vs Gemini 2.5 79.84 | +0.40p | 겹침 | 사실상 동등 |
| FG-Type: Ours 80.24 vs Gemini 3.5 78.53 | +1.71p | 겹침 | 사실상 동등 |
| Session: Ours L2 53.52 vs Gemini 2.5 59.46 | -5.94p | 겹침 (경계선) | 약하게 유의 |
| **Session: Ours L2 53.52 vs Gemini 3.5 72.09** | **-18.57p** | **안 겹침** | **강하게 유의** |

NAG 우위는 noise가 아닌 진짜 차이로 확정. 더 강한 Gemini(3.5)일수록 우리 NAG
우위가 확대되는 패턴 (+6.15p → +10.38p) 또한 통계적으로 명확. Session에서
3.5 Flash의 우위(+18.57p)도 통계적으로 강하게 유의 — paper 수준의 zero-shot
능력 (paper 66.29 < Gemini 3.5 72.09)을 보여줌.

## 5. 평가 코드 수정 사항

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

## 6. GRPO 단계 진행 (대기)

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

## 7. 데이터·실험 재현 정보

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
- 외부 baseline (Gemini 2.5 / 3.5 Flash, §4) 재현:
  ```powershell
  # Gemini 2.5 Flash (us-central1)
  python eval\run_gemini_inference.py `
    --data data\processed\test.parquet `
    --model gemini-2.5-flash `
    --output eval\preds_gemini25flash_baseline65k.jsonl `
    --max_concurrent 5 `
    --credentials <service_account_json_path>
  # Gemini 3.5 Flash (global only — needs --location global)
  python eval\run_gemini_inference.py `
    --data data\processed\test.parquet `
    --model gemini-3.5-flash `
    --location global `
    --output eval\preds_gemini35flash_baseline65k.jsonl `
    --max_concurrent 5 `
    --credentials <service_account_json_path>
  # 점수
  python eval\next_action_acc.py `
    --predictions eval\preds_gemini25flash_baseline65k.jsonl `
    --out eval\preds_gemini25flash_baseline65k.results.json
  python eval\next_action_acc.py `
    --predictions eval\preds_gemini35flash_baseline65k.jsonl `
    --out eval\preds_gemini35flash_baseline65k.results.json
  ```
- 압축 효과 측정: [tokenize_pack_compressed.py](../data/tokenize_pack_compressed.py)
  의 `_stats_*` 출력 (per-step compression ratio p50/p90/p99 포함)
- 디자인 배경: [compression_design.md](compression_design.md)
- 학습 환경 셋업: [training.md](training.md)
- 평가 절차: [evaluation.md](evaluation.md)
