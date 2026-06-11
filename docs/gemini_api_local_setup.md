# Gemini API 로컬 PC 호출 가이드 (Samsung 사내망)

Samsung 사내망 환경에서 로컬 Windows PC로 Google Vertex AI Gemini API를
호출하기 위한 셋업 가이드. PowerShell + Python 환경 가정.

다음을 다룹니다:
1. 사전 요구사항 (Python, 서비스 계정 JSON, 사내 프록시)
2. 환경 설정 (proxy + GCP credentials)
3. 예제 스크립트 ([scripts/gemini_text_api_test.py](../scripts/gemini_text_api_test.py))
4. 모델별 location 차이 (gemini-2.5-flash vs gemini-3.5-flash)
5. 자주 발생하는 에러와 해결법

---

## 1. 사전 요구사항

### 1-1. Python 환경

- Python 3.10 이상 (Windows 64-bit)
- pip 사용 가능

필요 패키지 설치:
```powershell
pip install google-cloud-aiplatform
```

(`vertexai` 패키지는 `google-cloud-aiplatform`에 포함됨)

### 1-2. GCP 서비스 계정 JSON

Vertex AI 사용 권한이 있는 GCP 프로젝트의 서비스 계정 JSON 파일이 필요합니다.

발급 절차:
1. GCP Console → IAM & Admin → Service Accounts
2. 사용할 서비스 계정 선택 → Keys 탭
3. ADD KEY → Create new key → JSON
4. 다운로드된 JSON 파일을 안전한 폴더에 보관

> 참고: 서비스 계정에 **"Vertex AI User"** 역할(Role)이 부여되어 있어야
> Gemini 모델 호출이 가능합니다.

JSON 파일 경로 예시:
```
C:\Users\<USERNAME>\Documents\gemini_api\<project-name>-<random>.json
```

### 1-3. 사내 프록시 정보

Samsung 사내망에서 외부 인터넷(googleapis.com 등)에 접근하려면 사내 프록시를
거쳐야 합니다.

| 항목 | 값 |
|---|---|
| 프록시 주소 | `http://10.244.254.254:8080` |
| 우회 대상(NO_PROXY) | `localhost,127.0.0.1,.samsung.net,.sec.samsung.net` |

---

## 2. 환경 설정

PowerShell에서 두 가지 환경 변수를 설정해야 합니다.

### 2-1. 임시 (현재 세션만)

PowerShell 창마다 매번 입력:

```powershell
# 사내 프록시
$env:HTTP_PROXY  = "http://10.244.254.254:8080"
$env:HTTPS_PROXY = "http://10.244.254.254:8080"
$env:NO_PROXY    = "localhost,127.0.0.1,.samsung.net,.sec.samsung.net"

# GCP 서비스 계정 (본인 경로로 교체)
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\Users\<USERNAME>\Documents\gemini_api\<your-key>.json"
```

장점: 간단하고 환경을 더럽히지 않음
단점: PowerShell 창 닫으면 사라짐 → 매번 입력

### 2-2. 영구 등록 (사용자 환경 변수)

한 번만 실행하면 이후 모든 PowerShell 창에서 자동 적용:

```powershell
[System.Environment]::SetEnvironmentVariable("HTTP_PROXY",  "http://10.244.254.254:8080", "User")
[System.Environment]::SetEnvironmentVariable("HTTPS_PROXY", "http://10.244.254.254:8080", "User")
[System.Environment]::SetEnvironmentVariable("NO_PROXY",    "localhost,127.0.0.1,.samsung.net,.sec.samsung.net", "User")
[System.Environment]::SetEnvironmentVariable("GOOGLE_APPLICATION_CREDENTIALS", "C:\Users\<USERNAME>\Documents\gemini_api\<your-key>.json", "User")
```

영구 등록 확인:
```powershell
# 새 PowerShell 창 열고 확인
$env:HTTP_PROXY
$env:GOOGLE_APPLICATION_CREDENTIALS
```

값이 정상 출력되면 등록 완료.

> ⚠ 주의: 영구 등록 시 **사외(집/카페) 환경에서는** PROXY가 살아있어 모든
> 외부 HTTPS 호출이 실패합니다. 사외에서 작업할 때는 임시로
> `$env:HTTP_PROXY=""; $env:HTTPS_PROXY=""`로 비우거나, 영구 등록을
> 해제하세요:
> ```powershell
> [System.Environment]::SetEnvironmentVariable("HTTP_PROXY",  $null, "User")
> [System.Environment]::SetEnvironmentVariable("HTTPS_PROXY", $null, "User")
> ```

### 2-3. SSL 인증서

별도 설정 불필요. 기본 Python 설치본의 `certifi` CA bundle만으로 정상 작동
하는 것을 확인했습니다 (Samsung 프록시가 Vertex AI 도메인에 대해서는 SSL
inspection을 하지 않는 것으로 추정).

> 만약 SSL 에러가 발생하면 §5-A 참조.

---

## 3. 예제 스크립트 — 첫 호출 테스트

[scripts/gemini_text_api_test.py](../scripts/gemini_text_api_test.py)를 사용합니다.

### 3-1. 실행

```powershell
# 위 §2의 환경 변수가 설정된 PowerShell 창에서
cd <repo-root>
python scripts/gemini_text_api_test.py
```

기본 동작:
- 모델: `gemini-2.5-flash`
- Location: `us-central1`
- Prompt: "한 문장으로 자기소개를 해줘."
- 출력: 콘솔 + `gemini_text_output.json` 파일

### 3-2. 다른 모델/프롬프트 시도

```powershell
# Pro 모델
python scripts/gemini_text_api_test.py --model gemini-2.5-pro

# 3.5 Flash (global location 필수, §4 참조)
python scripts/gemini_text_api_test.py --model gemini-3.5-flash --location global

# 사용자 prompt
python scripts/gemini_text_api_test.py --prompt "오늘 날씨를 시 형식으로 써줘"

# 서비스 계정 JSON 경로 명시
python scripts/gemini_text_api_test.py --credentials "C:\path\to\your-key.json"
```

### 3-3. 성공 시 출력 예시

```
Vertex AI 초기화 중...
gemini-2.5-flash 모델 로드 중...
텍스트 생성 요청을 보냅니다...
성공! 응답:
저는 Google의 대규모 언어 모델로, 텍스트 생성과 질문에 답하기 위해 만들어졌습니다.
결과를 'gemini_text_output.json'에 저장했습니다.
```

---

## 4. 모델별 location 차이

Vertex AI는 모델마다 사용 가능한 region이 다릅니다. **신규/preview 모델은
종종 `global` location에서만 호출 가능**합니다.

확인된 가용 model ID와 location:

| 모델 ID | Location | 비고 |
|---|---|---|
| `gemini-2.5-flash` | `us-central1` | 안정, 가장 저렴 |
| `gemini-2.5-pro` | `us-central1` | 고성능 |
| `gemini-3.5-flash` | **`global`** (us-central1엔 없음) | 최신 Flash. global만 가능 |

(Gemini 3.0/3.1 변종은 현재 확인되지 않음 — 본 문서 작성 시점 기준)

### 사용자 환경에서 직접 확인하는 방법

```powershell
python -c "
import os, vertexai
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\path\to\key.json'
vertexai.init(project='YOUR_PROJECT_ID', location='us-central1')
from vertexai.generative_models import GenerativeModel
for m in ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-3.5-flash']:
    try:
        r = GenerativeModel(m).generate_content('hi')
        print(f'OK: {m}')
    except Exception as e:
        print(f'FAIL: {m} -> {type(e).__name__}: {str(e)[:80]}')
"
```

`FAIL`이 뜨는 모델은 `--location global`로 재시도하면 잡힐 수 있습니다.

---

## 5. 자주 발생하는 에러와 해결법

### A. `ssl.SSLError: CERTIFICATE_VERIFY_FAILED`

Samsung 프록시가 SSL inspection 모드일 때 발생.

해결책 (우선순위 순):
1. **Samsung CA 인증서를 certifi에 추가**:
   - Samsung IT에서 회사 CA bundle 파일 받음
   - `python -c "import certifi; print(certifi.where())"`로 cacert.pem 경로 확인
   - 회사 CA를 cacert.pem 끝에 append
2. **환경 변수로 별도 bundle 지정**:
   ```powershell
   $env:REQUESTS_CA_BUNDLE = "C:\path\to\samsung-ca.pem"
   $env:SSL_CERT_FILE      = "C:\path\to\samsung-ca.pem"
   ```
3. **(비추천) SSL 검증 끄기**:
   ```powershell
   $env:PYTHONHTTPSVERIFY = "0"
   ```

### B. `Connection refused` / `timeout`

프록시 미설정 또는 잘못된 주소.

확인:
```powershell
# 환경 변수 확인
$env:HTTP_PROXY
$env:HTTPS_PROXY

# 프록시 서버 접근 가능한지
Test-NetConnection -ComputerName 10.244.254.254 -Port 8080
```

`TcpTestSucceeded: False`면 사외망이거나 프록시 주소가 변경된 것 — IT 팀
문의.

### C. `Unable to find your project. Please provide a project ID`

`GOOGLE_APPLICATION_CREDENTIALS`가 가리키는 JSON 파일을 찾지 못하거나,
JSON 안의 `project_id` 필드 미설정.

해결:
```powershell
# 파일 존재 확인
Test-Path $env:GOOGLE_APPLICATION_CREDENTIALS

# JSON 안의 project_id 확인
Get-Content $env:GOOGLE_APPLICATION_CREDENTIALS | ConvertFrom-Json | Select-Object project_id
```

값이 비었으면 서비스 계정 JSON을 다시 발급받거나 `vertexai.init(project=...)`로
직접 지정.

### D. `404 Publisher Model ... was not found`

요청한 모델 ID가 해당 location에 없음. §4 참조해서 location 변경:

```powershell
# 예: 3.5 Flash는 global에만 있음
python scripts/gemini_text_api_test.py --model gemini-3.5-flash --location global
```

### E. `PermissionDenied: 403 ... Vertex AI User`

서비스 계정에 Vertex AI 호출 권한 부족. GCP Console에서:
1. IAM & Admin → 해당 서비스 계정 검색
2. "Edit principal" → "Vertex AI User" 역할 추가

### F. `quota exceeded` / `RESOURCE_EXHAUSTED`

API rate limit 도달. 동시 호출 줄이고 backoff:
```powershell
# 잠깐 기다린 후 재시도
Start-Sleep -Seconds 30
python scripts/gemini_text_api_test.py
```

지속되면 GCP Console → Quotas에서 사용량 확인.

### G. `MemoryError` (큰 prompt 전송 시)

8 GB 이하 RAM에서 1 MB+ prompt 전송 시 가끔 발생. 다른 앱 종료 후 재시도.

---

## 6. 셀프 진단 명령

문제 발생 시 한 번에 환경 체크:

```powershell
Write-Host "=== Env vars ===" -ForegroundColor Cyan
"HTTP_PROXY  = $env:HTTP_PROXY"
"HTTPS_PROXY = $env:HTTPS_PROXY"
"NO_PROXY    = $env:NO_PROXY"
"GOOGLE_APPLICATION_CREDENTIALS = $env:GOOGLE_APPLICATION_CREDENTIALS"
"Credentials file exists: $(Test-Path $env:GOOGLE_APPLICATION_CREDENTIALS)"

Write-Host "`n=== Network ===" -ForegroundColor Cyan
Test-NetConnection -ComputerName 10.244.254.254 -Port 8080 -InformationLevel Quiet | ForEach-Object { "Proxy reachable: $_" }

Write-Host "`n=== Python ===" -ForegroundColor Cyan
python -c "import sys; print(sys.version); import vertexai; print('vertexai:', vertexai.__version__)"

Write-Host "`n=== Direct API test ===" -ForegroundColor Cyan
python -c @"
import urllib.request, ssl
try:
    r = urllib.request.urlopen('https://aiplatform.googleapis.com/', timeout=10)
    print('OK status:', r.status)
except Exception as e:
    print(type(e).__name__, ':', str(e)[:150])
"@
```

세 섹션 모두 정상이면 gemini_text_api_test.py가 작동할 환경입니다.

---

## 7. 부록 — 영구 등록 해제

영구 등록한 환경 변수를 모두 제거:

```powershell
[System.Environment]::SetEnvironmentVariable("HTTP_PROXY",  $null, "User")
[System.Environment]::SetEnvironmentVariable("HTTPS_PROXY", $null, "User")
[System.Environment]::SetEnvironmentVariable("NO_PROXY",    $null, "User")
[System.Environment]::SetEnvironmentVariable("GOOGLE_APPLICATION_CREDENTIALS", $null, "User")
```

새 PowerShell 창에서 확인 시 비어있으면 해제 완료.

---

## 관련 문서

- 본 문서는 외부 API 호출용 환경 셋업만 다룹니다.
- Customer-R1 본 실험 환경 (서버 학습/평가): [training.md](training.md)
- Gemini 결과를 사용한 외부 baseline 비교: [sft_results.md](sft_results.md) §4
