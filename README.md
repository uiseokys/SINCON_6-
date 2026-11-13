# SINCON 2026 제출 자료 — 팀 핫식스

## 1. 최종 예측 파일

| | 파일 (`predictions/`) | SHA-256 | Public | 구성 |
|---|---|---|---:|---|
| 최종 예측 | `submission_final_main.csv` | `aa455744556514d5…` | 0.143043 | 1~3단계 + 타 노선 층(주성분 30개) + 시간대 층. Kaggle에 올린 이름은 `submission_rail_factors_k30_hour_on_xgb7.csv` |

Kaggle에서 최종으로 선택한 제출 파일과 내용이 같고(바이트 동일) 이름만 바꿨다. 전체 해시는 `predictions/SHA256SUMS.txt`. Public 점수는 팀장이 Kaggle에서 확인한 값이다.

## 2. 폴더 구성

| 경로 | 내용 |
|---|---|
| `README.md` | 이 문서 (개요, 실행 환경과 방법, 자체 점검) |
| `SINCON_full_pipeline.ipynb` | **전체 실행 노트북.** 1단계부터 최종 예측까지 차례로 실행하고 제출 파일과 비교한다. 2026-11-13에 실행한 결과가 들어 있다 |
| `predictions/` | 최종 예측 원본 1개 |
| `code/requirements.txt` | 고정한 의존성 |
| `code/1_base_models/` | **1단계.** 기본 모델 학습·결합·하위 보정. `SINCON_final.ipynb`, `pipeline.py`, `full_train.py`, `full_stage.py`, 모델 코드와 입력은 `payload/` |
| `code/2_corrections/` | **2단계.** 보정 6개. `SINCON_final_step2_layers.ipynb`, `final_layers.py` |
| `code/2b_monday/`, `code/2c_cup_walk/` | 3단계가 읽는 두 구성 입력을 만드는 코드 |
| `code/3_stack/` | **3단계.** 보정 7개 단계. `SINCON_final_stack.ipynb`, `stack_*.py`. 모델 학습 코드 `training_scripts/`, 과거 예측 생성 코드 `oof_generation/` |
| `code/4_rail/` | **4단계.** 다른 노선 자료를 쓰는 층. `SINCON_rail_factors.ipynb`, `factor_layer.py`, `hol_layer.py`, `hour_layer.py`. 자료 수집 `fetch/`, 모델 학습 `training/`, 계수 검증 `validation/` |
| `docs/METHOD.md` | 방법 설명서 (검증 분할, 변수, 모델, 계수 결정, 제출한 후보) |
| `docs/DATA_AND_ARTIFACTS.md` | 생성 경로, 데이터 출처 작성표, 모델·중간 산출물 작성표 |
| `docs/DISCLOSURE.md` | Public 점수 사용, 평가 기간 자료의 사용, 규칙 관련 경과 |
| `docs/RETRAINING.md` | 보관 예측을 처음부터 다시 학습하는 절차와 시드 |
| `docs/sources/` | 외부 자료 출처·조회일·이용 조건 원표 3개 |
| `records/executed_runs/` | 전체 실행 노트북을 2026-11-13에 실행했을 때의 단계별 로그, 요약(`run_summary.json`), `pip freeze` |
| `records/development_history.zip` | 기존 개발 기록 묶음 (작업 일지, 사전 기록, 검증·감사 문서, 탈락한 실험 스크립트, 패키지를 만들 때의 검증 로그) |

- 실행과 재학습에 필요한 코드·자료만 `code/` 에 남겼다. 그 밖의 기록은 지우지 않고 `records/development_history.zip` 하나에 모았다.
- 노트북 안의 설명에는 개발 때의 패키지 이름이 남아 있다: `final_package` = `1_base_models`, `final_delivery_v2` = `2_corrections`, `monday_delivery` = `2b_monday`, `final_stack_delivery` = `3_stack`.

## 3. 실행 방법

### 환경 준비

| 항목 | 내용 |
|---|---|
| OS·기기 | macOS 26.6 (arm64), Apple M5 Max, 메모리 64GB. Linux에서의 실행은 확인하지 못했다 |
| 언어 | Python 3.12.15 (1단계 노트북은 3.11과 3.12만 받는다) |
| 의존성 | `code/requirements.txt`: numpy 2.5.4, pandas 3.1.1, scipy 1.17.1, scikit-learn 1.6.1, lightgbm 4.6.0, torch 2.9.1, xgboost 3.4.1, tabicl 2.2.0, einops 0.8.2, huggingface-hub 0.36.0, psutil 7.2.2, tqdm 4.70.1. 실행에 쓴 환경 전체는 `records/executed_runs/pip_freeze_python312.txt` |
| 설치 | `python3.12 -m venv .venv && . .venv/bin/activate && pip install -r code/requirements.txt nbclient nbformat nbconvert ipykernel` |
| 커널 등록 | `python -m ipykernel install --sys-prefix --name sincon312` 을 실행하고, 아래 명령 앞에 `JUPYTER_PATH="$VIRTUAL_ENV/share/jupyter"` 를 붙인다. 등록하지 않으면 기본 이름 `python3` 이 시스템의 다른 Python 커널로 연결될 수 있다 |
| CPU·GPU | 조립 경로는 CPU만 쓴다. 재학습은 CPU로 되고, CUDA가 있으면 1단계 신경망이 쓴다(실행해 보지는 못했다) |

### 학습 및 선택

| 항목 | 내용 |
|---|---|
| 실행 명령 | `docs/RETRAINING.md`: 1단계 전체 재학습(1절), 3단계의 모델 6개(2절), 4단계의 자료 수집과 XGBoost 2개(3절), 과거 예측(4절) |
| 허용 초기 모델 | TabICL 2.2.0 회귀 체크포인트 `tabicl-regressor-v2-20260212.ckpt` (공개 사전학습 모델, BSD-3-Clause, `code/1_base_models/payload/models/`). 그 밖의 모델은 `train.csv` 로 처음부터 학습한다 |
| 검증 분할 | 시간 순서 분할, 기준일 9개(2025-06-30 ~ 2026-04-30). `docs/METHOD.md` 2절 |
| 설정·시드 | `docs/METHOD.md` 3절, `docs/RETRAINING.md` 의 "시드와 환경" |
| 생성 모델 | 쓰지 않았다(해당 없음) |
| 예상 시간 | 1단계 전체 재학습: 로컬 CPU에서 약 5.5시간. 3·4단계 모델: 재지 않았다 |

### 최종 추론

| 항목 | 내용 |
|---|---|
| 실행 명령 | `SINCON_full_pipeline.ipynb` 를 열어 모두 실행한다. 명령으로는 `JUPYTER_PATH="$VIRTUAL_ENV/share/jupyter" jupyter nbconvert --to notebook --execute SINCON_full_pipeline.ipynb --output executed.ipynb --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=sincon312` |
| 하는 일 | 1 → 2 → 2b → 2c → 3 → 4 순서로 각 단계 노트북의 코드 셀을 별도 프로세스에서 실행하고, 최종 예측을 `predictions/` 의 제출 파일과 비교한다. 수치 차이가 1e-9를 넘으면 실패로 끝난다 |
| 모델·입력 경로 | 보관한 모델 예측과 패키지 안의 자료(`docs/DATA_AND_ARTIFACTS.md` 3절). 경로는 스크립트가 지정한다 |
| 후처리 설정 | 각 노트북의 설정 셀과 층 모듈의 상수. 값은 `docs/METHOD.md` 4~6절 |
| 출력 | `run_outputs/submission_final_main.csv` (4단계 출력 `run_outputs/rail/submission_rail_factors_k30_hour_on_xgb7.csv` 를 복사한 것), `run_outputs/run_summary.json`, `run_outputs/logs/` |
| 걸리는 시간 | 약 42초 (1단계 29초, 나머지 13초) |

- 단계별 노트북을 하나씩 실행해도 된다. 2·3·4단계 노트북 첫 설정 셀의 `BASELINE_CSV` 에 앞 단계 출력의 경로를 적는다. `2_corrections` 와 `2b_monday` 는 `data/` 에 제공 CSV 세 개가 있어야 한다(전체 실행 노트북이 1단계 폴더의 사본을 복사한다).
- 1단계 노트북 첫 셀은 `pip install` 을 실행한다. 인터넷이 없으면 `SINCON_PIP_NO_INDEX=1` 을 주고 미리 설치해 둔다.
- 1단계와 2단계는 패키지 안의 `outputs/` 폴더에 결과를 쓴다.

### 재현 변동

| 항목 | 내용 |
|---|---|
| 조립 경로 | 난수를 쓰지 않는 결정적 계산이다. 2026-11-13에 이 폴더의 코드를 새 Python 3.12.15 환경에서 1단계부터 이어 실행한 결과, 최종 예측은 제출 파일과 **최대 log1p 차이 8.9e-16** 이었다. 바이트는 달랐다 |
| 바이트가 달라지는 곳 | 1단계 출력은 제출 때의 파일과 바이트까지 같았다. 2단계를 다시 실행한 출력이 끝자리에서 달라(8.9e-16) 그 뒤 CSV의 바이트가 달라진다. 4단계를 패키지에 든 3단계 기준 파일에서 시작하면 최종 예측이 제출 파일과 바이트까지 같다 |
| 버전에 따른 차이 | 3·4단계를 보관한 주후보에서 시작했을 때: Python 3.12.15·3.11.17(pandas 3.1.1)은 바이트 일치, pandas 2.2.2·numpy 2.0.2와 Python 3.14.6(pandas 3.0.6)은 수치 일치(최대 1.8e-15)이고 바이트는 다름. 정리하기 전의 패키지로 확인한 결과다 |
| 1단계 전체 재학습 | 비결정적이다. 같은 시드라도 신경망은 기기·라이브러리에 따라 달라진다. 로컬 CPU 재학습 1회의 결과는 제출 기준 파일과 평균 절대 log 차이 0.0012, RMS 0.0027, 최대 0.153이었고 99.85%의 행이 0.01 이내였다. 검증 점수는 세 구간 모두 0.00005 이내로 같았다 |
| 재학습 결과를 이어 붙일 때 | 2단계는 입력이 기준 파일과 1e-9 이내로 같을 때만 진행한다. 재학습한 1단계 출력은 이 검사에서 멈춘다. 재학습 결과로 끝까지 가려면 그 검사를 풀어야 하고, 그렇게 만든 최종 예측의 점수는 재 보지 않았다 |
| 3·4단계의 보관 모델 | XGBoost·LightGBM은 시드 0 고정. MLP 4시드는 Apple MPS에서 학습한 보관본이라 다른 기기에서 값이 달라진다. 다시 학습해 보관본과 대조하지는 않았다 |
| 반복 실행 결과 | `records/executed_runs/`. 패키지를 만들 때의 검증 기록은 `records/development_history.zip` |

## 4. 제출 자료 목록과 위치

| 안내서 항목 | 위치 |
|---|---|
| 최종 예측 파일 | `predictions/` |
| 전체 실행 코드와 실행 순서 | `SINCON_full_pipeline.ipynb` (실행 결과 포함)와 `code/` |
| 데이터 출처 및 생성 기록 | `docs/DATA_AND_ARTIFACTS.md` 1~2절, `docs/sources/` |
| 실행 환경 및 방법 | 이 문서 3절, `code/requirements.txt`, `records/executed_runs/pip_freeze_python312.txt` |
| 설명서 | `docs/METHOD.md` |
| 모델 및 중간 산출물 | `docs/DATA_AND_ARTIFACTS.md` 3절 (파일은 각 단계 폴더 안), 재생성 방법은 `docs/RETRAINING.md` |
| 기존 개발 기록 | `records/development_history.zip` |

## 5. 제출 전 자체 점검

- [x] 최종 예측 파일과 실제 사용한 코드, 모델, 설정의 관계를 명시했다 (1절, 3절, `docs/DATA_AND_ARTIFACTS.md` 1절).
- [x] 데이터와 모델 작성표에 사용 항목을 기록하고 미사용 항목은 해당 없음으로 표시했다 (`docs/DATA_AND_ARTIFACTS.md`).
- [x] 제공 자료부터 최종 예측까지의 경로, 명령, 의존성, 시드를 설명했다 (3절, `docs/RETRAINING.md`). 순방향 과거 예측과 3·4단계 보관 모델은 이번에 다시 만들어 대조하지 못했다.
- [x] 공개 데이터에서 평가 관련 자료를 열람·확보·사용한 사실을 기재했다 (`docs/DISCLOSURE.md`).

## 6. 그 밖에 알아 둘 점

- 일부 스크립트와 매니페스트에는 개발 기기의 절대 경로가 남아 있다. `records/executed_runs/` 의 실행 결과에서는 실행 폴더 경로를 `<실행 폴더>`, 가상환경 경로를 `<가상환경>` 으로 바꿨다(그 밖의 내용은 그대로다).
- `code/3_stack/oof_generation/`, `code/4_rail/training/`, `code/4_rail/validation/` 의 스크립트는 원 작업 폴더의 경로를 가정한다. 이 폴더 안에서 그대로 실행되도록 손보지 않았다.
- 비밀번호, API 키, 토큰은 들어 있지 않다.
