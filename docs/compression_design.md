# Customer-R1 Context Compression — Design Doc

작성: 2026-05-26 · 상태: 설계 단계 (측정 데이터 도착 전)

## 1. Motivation

Customer-R1 paper(arxiv 2510.07230)는 next-action 예측을 위해 prompt에
`system + persona + (obs_1, action_1, rationale_1, ..., obs_t-1, action_t-1, rationale_t-1) + obs_t`
를 넣어 65K context, Qwen2.5-7B-Instruct-1M, batch 64로 학습한다.
SFT 2,000 steps + GRPO 2 epoch — A100 64장급 연산을 가정한다.

**가설:** Amazon.com 페이지는 step간 redundancy가 크다(nav_bar, footer, sidebar는
세션 내내 거의 동일). paper의 truncation 알고리즘은 "가장 오래된 step의 HTML을
통째로 폐기"하는 단순 방식이라 압축 헤드룸이 크게 남아 있다. **같은 65K budget
안에서 더 많은 의미 있는 step을 담으면 next-action 정확도가 같거나 오히려
향상될 수 있다.**

증명되면 두 가지 이득:
- **연산 절감**: 동일 정확도를 더 작은 모델·짧은 context로 달성 가능
- **정확도 향상**: 동일 자원에서 더 많은 history step이 모델에 노출

## 2. Baseline (paper) 분석

`data/tokenize_pack.py` 의 핵심 알고리즘 (paper §4.1):

```
1. history step 각각의 simplified_html을 그대로 user.jinja에 채워 prompt 생성
2. prompt token 수가 65,000 초과면, history 가장 오래된 step의 HTML을
   "[earlier page HTML omitted to fit context window]" 마커로 교체
3. 여전히 초과면 다음 오래된 step도 같은 마커로 교체... (oldest-drop)
4. 모든 history HTML이 omitted 되어도 초과면, current step HTML을 절반씩 자름
```

**구조적 약점:**
- HTML 단위 all-or-nothing: 한 step의 HTML은 통째로 유지 또는 통째로 폐기
- 같은 페이지의 nav_bar / footer 가 매 step마다 풀 텍스트로 반복
- step간 차분(diff)가 작은데도 전체를 그대로 넣음
- 가장 오래된 step의 정보는 완전히 사라짐 (action / rationale 제외)

## 3. 압축 레이어 설계 (적용 순서)

> **원칙:** 정보 손실 0 → 작음 → 큼 의 순서로 점진 적용. 단계마다 비교 평가
> 가능하도록 hook을 둔다 (`--compression {none, L1, L1L2, L1L2L3}`).

### Layer 1 — Static Furniture Deduplication (정보 손실 0)

**아이디어:** 같은 session 안에서 모든 step의 HTML에 substring으로 등장하는
구역(=고정 chrome)을 식별하고, 첫 등장 후 짧은 reference로 대체한다.

**알고리즘 (개요):**
1. 세션의 모든 step HTML을 입력으로 받음
2. 일정 길이(예: 200 chars) 이상의 모든-step 공통 substring을 그리디 추출
3. 가장 큰 것부터 ID 부여 (`F1, F2, ...`)
4. 각 step에서 furniture 영역을 `[[F1]]`, `[[F2]]` 토큰으로 교체
5. prompt 끝에 `# Furniture` 섹션을 한 번만 정의:
   ```
   # Furniture (persistent across all pages)
   F1: <div name="nav_bar">...</div>
   F2: <div name="footer">...</div>
   ```

**정보 손실:** 이론상 0 (모델이 `[[F1]]` 의미를 학습하면 가역). 다만 명시적
furniture 섹션을 모델이 활용하도록 user.jinja에서 명확히 가이드 필요.

**기대 절감:** 30~50% (가설, Compression 1단계 측정으로 확정).

### Layer 2 — Action-Anchored History Subtree (정보 손실 작음)

**아이디어:** history step의 HTML은 그 step에서 **실제 action이 발생한
element + 그 직계 조상/형제**만 보존하고, 나머지 영역은 짧은 marker로 대체.

이유: 모델이 history에서 알아야 할 것은 "이 페이지에서 어디를 클릭했고
어떤 페이지로 갔는가" — 클릭하지 않은 다른 element들의 디테일은 거의 무관.
현재 step(예측 대상)의 HTML은 풀 보존.

**알고리즘:**
1. history step의 `action_wire_json` 에서 `name` 추출 (action target element ID)
2. simplified_html에서 그 element를 찾음 (`name="<id>"` attribute로)
3. 그 element의 ancestor chain + immediate siblings 보존
4. 나머지 형제·subtree는 `<region name="..." elided="N elements"/>` 마커
5. 단, `terminate` action은 anchor가 없으므로 전체 보존(또는 페이지 상단 일부만)

**정보 손실:** 작음. 잠재적 위험은 모델이 "그 페이지에 다른 무엇이 있었기에
이 element를 클릭했다" 같은 contextual reasoning을 잃는 것. 측정으로 검증.

**기대 추가 절감:** 50~70% (Layer 1 위에 적용).

### Layer 3 — Budget-Aware Fallback (paper의 oldest-drop 유지)

L1+L2 적용 후에도 65K 초과면, paper의 oldest-drop을 잔여 보완책으로 사용.
다만 발동률은 거의 0이 되어야 정상.

### (선택) Layer 4 — Semantic Summary

훨씬 공격적: 각 history step을 Gemini로 1-2문장 요약으로 압축. 90%+ 절감
가능하지만 element name 등의 정확한 정보 손실. 학습 결과가 L1+L2로
만족스럽지 않을 때만 시도.

### 3.5 — Worked Example (실제 OPeRA 세션)

> 측정 코드: `scripts/build_example.py`
> 세션 ID: `4d439e8e-69d1-4ee2-8030-7c885c6b1fa2_2025-04-24...`
> 시나리오: 같은 product page에서 옵션 button(Dolphin → Narwhal → terminate)을 누르는 3-step 짧은 세션.

**Session 통계:**
| step | role | action | simplified_html chars |
|---|---|---|---|
| 0 | history | click `product_options.style.button_list.dolphin_6_99_6_99` | 58,127 |
| 1 | history | click `product_options.style.button_list.narwhal_6_99_with_28_percent_savings...` | 58,127 |
| 2 | **current** | terminate | 47,161 |
| | | **total** | **163,415** (~40K tokens) |

#### Baseline (paper 방식)

3개 step의 simplified_html을 그대로 prompt에 채움 → **163,415 chars (~40K tokens)**.
같은 product page라 nav_bar/footer/buybox 같은 영역이 매 step 거의 동일하게 반복.

#### Layer 1 — Static Furniture Dedup 적용

알고리즘이 자동으로 추출한 furniture 4개 (총 4,094 chars, 모든 step 공통):

| ID | chars | 영역 미리보기 |
|---|---|---|
| `[[F1]]` | 2,763 | `<html><head></head><body><div name="nav_bar" role="navigation" aria-label="Primary"><div name="nav_bar.homepage">Homepage</div>...` |
| `[[F2]]` | 689 | `<div name="buybox.purchase_form.quantity_selector"><label>Quantity:</label>...` |
| `[[F3]]` | 389 | `<a name="buybox.purchase_form.purchase_detail.see_more"...` |
| `[[F4]]` | 253 | `customerReviews">4.6 out of 5 stars 57 ratings</a>...` |

각 step에서 위 4영역이 `[[F1]]`...`[[F4]]` 마커로 대체:

| step | before | after | ratio |
|---|---|---|---|
| 0 | 58,127 | 54,057 | ×0.93 |
| 1 | 58,127 | 54,057 | ×0.93 |
| 2 | 47,161 | 43,091 | ×0.91 |
| Furniture 정의 (한 번) | — | 4,094 | |
| **L1 총합** | **163,415** | **155,299** | **5.0% 절감** |

**관찰**: 3-step 세션이라 furniture 정의 오버헤드(4,094 chars)가 절감의 상당 부분을
상쇄. 10-step 세션이라면 (9 step × 4,094 chars 절감) − (4,094 chars 정의)
= 약 32K chars 절감 → ~20% 절감. 세션이 길수록 효과 큼.

#### Layer 1 + Layer 2 — Action-Anchored History 적용

history step 0, 1은 action target element 주변 window(±600 chars)만 보존,
current step 2는 L1만 적용한 채 full 유지:

| step | role | L1만 | L1+L2 |
|---|---|---|---|
| 0 | history | 54,057 | 1,266 chars (×0.023, **97.7% 압축**) |
| 1 | history | 54,057 | 1,266 chars (×0.023, **97.7% 압축**) |
| 2 | current | 43,091 | 43,091 (그대로) |
| Furniture 정의 | | 4,094 | 4,094 |
| **L1+L2 총합** | **163,415** | | **49,717 chars** (**69.6% 절감**) |

**모델이 보는 step 0의 모양 (압축 후 1,266 chars 발췌):**

```html
<!-- 202 chars elided (head) -->ings" role="button" href="javascript:void(0)">
4.6 4.6 out of 5 stars</a><span name="reviews" aria-label="23 Reviews">(23)
</span></div><span class="product-price">Price: $6.99</span>
<div name="alternative_offer"></div><div name=""></div>
<div name="product_options" class="product-options">
  <div name="product_options.style">
    <div><span>Style:</span><span>Dolphin</span></div>
    <ul name="product_options.style.button_list" role="radiogroup">
      <li name="...sting_ray_6_99...">Sting Ray $6.99 ...</li>
      <li name="...dolphin_6_99_6_99">Dolphin $6.99 $6.99</li>    ← clicked
      <li name="...narwhal_6_99_with_28_percent...">Narwhal ...</li>
    </ul>
  </div>
</div>
<div name="about_this_item">...
<!-- 52,655 chars elided (tail) -->
```

핵심 의미가 유지된다:
- **action target** (`product_options.style.button_list.dolphin_6_99_6_99`)이 보존됨
- 그 옆의 **다른 옵션들** (`sting_ray`, `narwhal`)도 같은 button_list 안에 있어서 자연히 보존 — 비교/선택 reasoning 가능
- **product 가격, 별점, reviews 카운트** 등 의사결정 정보 포함
- 페이지의 chrome(nav_bar, footer 등)은 furniture로 빠짐 → 별도 한 번만 정의

**잃은 것:**
- `about_this_item` 본문의 상세 설명 (tail에 elided)
- 페이지 하단의 reviews 본문, sponsored products 등

이 손실이 next-action 예측에 의미 있는지는 학습 후 메트릭으로 검증 필요.
페이퍼 §3.2의 difficulty 분류에서 `product_option` click은 reward weight 10
(가벼운 case) — 즉 같은 페이지의 옵션 선택은 비교적 단순한 task라 history의
다른 영역 손실 영향이 작을 가능성이 큼.

**다른 click_type (예: nav_bar, purchase)에서도 L2가 안전한지는 §6 Risk
mitigation에서 `preserve_named_elements=True` 옵션으로 보강해야 한다.**

### 3.6 — 전체 데이터셋 측정 결과 (가설 검증)

> 측정 코드: `scripts/measure_redundancy.py`
> 입력: `data/trajectories/train.jsonl` (437 sessions, 4864 steps)
> 출력: `data/redundancy_train_summary.json`, `data/redundancy_train.csv`

#### 데이터셋 규모 (paper의 65K token budget ≈ 260K chars 대비)

| 항목 | min | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| 세션당 step 수 | 3 | 7 | 22 | 62 | **241** |
| step당 HTML chars | 2.8K | 103K | 310K | 459K | **626K** |
| 세션당 총 HTML | 139K | 878K | 3.4M | 9.5M | **14.5M** |

**시사점:**
- **p90 step은 단일 HTML만으로 65K budget을 초과** (310K chars ≈ 78K tokens > 65K)
- **p50 session도 전체 history 넣으면 budget의 3.4배 초과**
- 즉 paper baseline에서도 oldest-HTML-drop truncation이 거의 항상 발동, history step의 다수가 통째로 omit된 채 학습 중

#### 인접 step 중복률

`(LCP + LCS) / current_step_html` 비율 = 직전 step과 시작·끝부분이 얼마나 겹치는가

| 백분위 | 비율 | 해석 |
|---|---|---|
| p50 | 0.3 | 직전 step과 30% 이상 동일 |
| p90 | 2.0 | LCP·LCS가 같은 영역을 가리킬 만큼 거의 동일 HTML |
| p99 | 2.0 | 사실상 같은 페이지 |

**90% 이상의 step pair에서 절반 이상이 직전 step과 중복**. diff-based 압축의 효과가 크다는 강한 신호.

#### Furniture 추정 (sampling-based)

세션의 모든 step에 공통으로 등장하는 substring 길이 (위치 무관):

| 백분위 | 세션당 furniture chars |
|---|---|
| p50 | 1,626 |
| p90 | 17,000 |
| p99 | 110,583 |
| max | 209,359 |
| **train 전체 합** | **3,651,729** (≈ 900K tokens 절감 가능) |

#### 압축 헤드룸 — 보수적 lower bound

```
raw total chars:         683,654,687
dedup (LCP+LCS only):    407,090,045
saving:                       40.5%
```

**인접 step 앞·뒤 중복만 빼도 40.5% 절감.** 실제 L1 furniture(위치 무관)와 L2 anchor-slicing을 적용하면 이보다 훨씬 큰 효과 예상.

#### 가설 검증 종합

| 가설 | 검증 결과 |
|---|---|
| Paper의 truncation이 자주 발동 | ✅ **강함** — p90 step만으로도 budget 초과 |
| 인접 step 중복이 매우 크다 | ✅ **매우 강함** — p90이 2.0 (거의 동일 HTML) |
| 압축 헤드룸 충분 | ✅ **40.5% lower bound**, 실 효과는 더 클 것 |
| 세션 길이 의존성 | ✅ p50=7 / max=241로 폭 큼. 긴 세션일수록 L1 효과 ↑ |
| §3.5의 5% L1 절감은 특수 케이스인가 | ✅ 3-step 세션이라 furniture 정의 오버헤드가 큼. 평균 11.1 step / 일부 241 step 세션에서는 효과 매우 큼 |

**결론: 압축 연구는 진행할 가치 있음.** L1+L2의 실제 학습 비교에서 next-action accuracy가 baseline과 같거나 우위면, "같은 GPU 자원으로 더 많은 history 입력 → 페이퍼 baseline 우위" 명제가 성립한다.

## 4. Information Loss & 평가 방법

평가 메트릭 두 축:

**A. 표면 메트릭 (모델 학습 없이 측정 가능):**
- 압축률: `len(compressed_prompt) / len(baseline_prompt)`
- 토큰 절감률: `n_tokens(compressed) / n_tokens(baseline)`
- Action target reachability: 압축 후에도 next action의 `name` 이 prompt
  안에 substring으로 존재하는 비율 (모델이 copy 가능한가)
- Truncation rate: 65K 초과로 oldest-drop이 발동된 sample 비율

**B. 학습 후 메트릭:**
- Next-action accuracy (overall)
- Per-click_type accuracy (특히 nav_bar / footer 영역 click 비율 비교)
- Rationale BERTScore / ROUGE-L (보조)
- 같은 budget(예: 32K)으로 강제했을 때 baseline vs compressed 비교

## 5. 구현 계획

### 5.1 신규 파일

- `data/compress_html.py` — Layer별 함수 + 단위 테스트
- `data/tokenize_pack_compressed.py` — `tokenize_pack.py`의 fork. CLI 옵션
  `--compression {none, L1, L1L2}` 추가. `none`이면 기존과 동일.
- `prompts/user_compressed.jinja` — Furniture 섹션 추가 + history 표현 명세
- `tests/test_compress_html.py` — Layer 별 invariant 테스트
  (예: L1 적용 후 + reference 치환 = 원본 HTML, L2 적용 후 action name 보존)

### 5.2 데이터 산출물 — 변형별 별도 보관

**원칙:** baseline / L1 / L1L2 등 각 압축 변형마다 별도의 `processed_*/` 폴더에
parquet을 생성·보관한다. on-the-fly 압축(학습 시점에 변환)을 쓰지 않는 이유:

| 이유 | 설명 |
|---|---|
| tokenize_pack은 비싸다 | Qwen 토크나이저로 5,856 sample 처리 → 수십 분. 학습 step마다 재실행하면 GPU가 토크나이저를 기다리는 병목 |
| 학습 코드 변경 0 | `train/sft.py`는 `--data path` 인자만 다르게 받음. baseline vs compressed 비교가 동일 코드로 진행되어 신뢰성 ↑ |
| 재현성 | 압축 알고리즘을 나중에 수정해도 그 시점에 학습한 dataset이 그대로 보존됨 |
| 디스크 비용 작음 | 변형별 약 0.5~2GB, 4종 다 만들어도 ~6GB |

### 5.3 권장 디렉토리 구조

```
data/
├── trajectories/              # Phase 1 (모든 변형 공유, 한 번만 생성)
│   ├── train.jsonl              # 437 sessions
│   └── test.jsonl               # 90 sessions
├── trajectories_synth/        # Phase 2 (모든 변형 공유, 한 번만 생성)
│   ├── train.jsonl              # + rationale_synth
│   └── test.jsonl
├── processed/                 # Phase 3 — baseline (paper 그대로)
│   ├── train.parquet
│   ├── test.parquet
│   └── manifest.json
├── processed_L1/              # Layer 1 (furniture dedup)
│   ├── train.parquet
│   ├── test.parquet
│   └── manifest.json
└── processed_L1L2/            # Layer 1 + 2 (furniture + action-anchored)
    ├── train.parquet
    ├── test.parquet
    └── manifest.json
```

Phase 1·2 산출물은 모든 변형이 공유 — 합성 비용($150)을 다시 치를 필요 없음.
Phase 3만 변형별로 재실행.

### 5.4 manifest.json — 변형 추적

각 `processed_*/` 폴더에 함께 두어, 6개월 뒤에도 어떤 dataset인지 추적 가능:

```json
{
  "compression": "L1L2",
  "params": {
    "furniture_min_chars": 200,
    "anchor_depth": 2,
    "preserve_named_elements": true
  },
  "generator": {
    "script": "data/tokenize_pack_compressed.py",
    "git_sha": "<commit hash at generation>",
    "tokenizer": "Qwen/Qwen2.5-7B-Instruct-1M",
    "max_prompt_tokens": 65000,
    "created_utc": "2026-05-26T15:00:00Z"
  },
  "input": {
    "trajectories_synth": "data/trajectories_synth",
    "trajectories_synth_sha": "<hash of synth dir>"
  },
  "stats": {
    "samples": 5856,
    "mean_prompt_tokens": 21500,
    "max_prompt_tokens": 64320,
    "truncation_rate": 0.012,
    "rationale_human": 207,
    "rationale_synth": 5649
  }
}
```

### 5.5 학습·평가 사용 패턴

```bash
# Baseline 학습
bash scripts/launch.sh --gpus 8 --model 7b --stage sft \
  --data_dir data/processed

# Compressed L1L2 학습 (동일 코드, 다른 데이터)
bash scripts/launch.sh --gpus 8 --model 7b --stage sft \
  --data_dir data/processed_L1L2
```

모델·데이터 변형 매트릭스가 폴더 단위로 깔끔히 분리되어, 결과를 manifest와
체크포인트 쌍으로 1:1 추적 가능.

## 6. Risk & Mitigation

| Risk | 영향 | Mitigation |
|---|---|---|
| L1 furniture 추출 부정확 → nav_bar element 자체가 reference로 사라짐 | nav_bar click action 학습 신호 손실 | (a) furniture 추출 시 `name="..."` attribute 가진 모든 element는 항상 보존, (b) baseline vs L1의 nav_bar accuracy 별도 측정 |
| L2 anchor 영역만 보존 → 비교/선택 reasoning 손실 | "다른 product도 봤는데 이게 더 싸서 클릭" 같은 reasoning이 약화 | rationale 텍스트는 압축 안 함 (Gemini가 이미 압축된 추론). history rationale이 그 정보를 들고 있을 가능성 큼 |
| Furniture marker `[[F1]]` 가 token 측면에서 비효율 | 작은 영역에는 marker가 원본보다 클 수 있음 | 압축 결과 길이가 원본보다 짧을 때만 적용 (length check) |
| 압축 데이터로 학습한 모델을 baseline 평가에 쓰면 부정합 | A/B 비교 불성립 | 평가용 데이터셋도 같은 압축 옵션으로 별도 생성, 평가 시 일관성 보장 |
| 정보 손실이 너무 커서 next-action accuracy 떨어짐 | 가설 반증 | L1만으로도 의미 있는 결과면 publish 가치 있음 (L2 안 가도 OK). 점진 적용 + Ablation |

## 7. Open Questions (측정 결과 보고 결정)

- Static furniture가 정말 30%+ 차지하는가? (Compression 1단계 측정)
- 평균 session의 step 수와 step별 HTML 길이 분포는? (oldest-drop 발동 현황)
- furniture를 entity-level(`<div name="nav_bar">...</div>` 통째)로 추출할지,
  substring-level(어떤 길이든 공통 substring)로 추출할지
- L2의 anchor 정의: action element 자체 / +parent / +sibling — 어느 범위?
