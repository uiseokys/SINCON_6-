# 재학습 절차

제출 파일을 수치까지 재현하는 기본 경로는 보관한 모델 예측에서 조립하는 경로다(`README.md` 3절). 이 문서는 그 보관 예측을 **처음부터 다시 학습해 만드는 방법**을 단계별로 모은 것이다. 다시 학습하면 신경망 때문에 값이 달라진다(`README.md` "재현 변동").

## 1. 1단계 기본 모델 (`code/1_base_models/`)

한 번에 실행:

```sh
cd code/1_base_models
SINCON_RUN_MODE=full SINCON_DEVICE=cpu jupyter nbconvert --to notebook --execute SINCON_final.ipynb \
  --ExecutePreprocessor.timeout=86400 --ExecutePreprocessor.kernel_name=sincon312 --output SINCON_final.executed.ipynb
```

- 노트북의 `FULL_STAGES` 가 단계를 고른다: `native`(팀 LightGBM + MLP 30시드), `stacks`, `rolling`, `hourly_xgb`, `clean_oof`, `event_responses`, `tabicl`, `vector`. 단계별 명령은 `full_stage.py` 에 있다.
- 로컬 CPU 측정: 약 5.5시간(신경망 30시드 3.9시간, rolling 51분, 나머지 42분. 다른 작업과 CPU를 나눠 쓴 값).
- 아래는 패키지를 만들 때 적은 단계별 설명이다. `$repo` 는 `code/1_base_models/payload/work/`, `$team` 은 `code/1_base_models/payload/team/` 에 해당한다. 본문에 나오는 노트북·보고서 링크 중 일부는 이 제출 폴더에 없다(개발 기록 묶음에 있다).

### 공식 train.csv부터 학습 순서

1. [로컬 모델 노트북](sincon_reproduce_local.ipynb)과 [조건별 결합 노트북](sincon_reproduce_stack.ipynb)에 원본 CSV 폴더를 지정한다. 전자는 `sincon_local_repro.zip`, 후자는 `sincon_conditional_repro.zip`을 사용한다. 두 노트북에서 `REBUILD_TEAM=False`를 유지한다. 이 경로는 팀의 보정 전 신경망을 불러오고 LightGBM, 그래프, 가법 시간대 모형, 버스·경기 모형, 수요 혼합, 축제·주거 보정을 공식 과거 자료에서 재학습한다. `submission_local_stack.csv`, `submission_conditional_stack.csv`, `sincon_local_replay_validation.csv.gz`를 얻는다.
2. 팀의 기본 LightGBM·MLP까지 재학습하려면 복사한 팀 코드와 동결된 `external/`로 `python "$repo/sincon_clean_native_forecasts.py" "$team" "$fresh_team" --threads 6`을 실행한다. 생성한 `ensemble_parts.csv`와 `valid_predictions_ABC.csv.gz`를 새로 푼 두 노트북의 복사본 팀 폴더에 각각 넣고 1단계를 실행한다. 원래 노트북의 선택적 `sincon_rebuild_team.py`는 옛 수준 보정을 출력하므로 이 경로에서 실행하지 않는다. GPU·CPU와 PyTorch 버전에 따라 새 MLP 수치가 조금 달라질 수 있다.
3. 과거 기준일 예측을 다시 학습하려면 `sincon_rolling_forecasts.py`에 로컬 노트북의 모델 인수 10개와 새 출력 폴더를 넘긴다. 2025-06-30, 07-31, 08-31, 09-30은 기본 설정이다. 2026년 3·4월 기준일에는 `--window 2026-03-31:2026-06-30 --window 2026-04-30:2026-06-30`을 사용한다. 엄격한 2025년 여름 비교는 같은 모델 인수를 `sincon_summer_replay.py`에 넘기고 `--seeds 30`으로 실행한다. 2026년 보조 두 개는 `sincon_target_season_validation.zip`에 보관했다.
4. 시간별 XGBoost는 `python "$repo/sincon_zoo_transfer_season.py" "$team" "$xgb" --origin 2026-06-30 --end 2026-10-31 --competition-test`로 재학습한다. 같은 명령에서 기준일과 종료일을 과거 검증에 맞추고 `--competition-test`를 빼면 과거 예측을 만든다. A/B/C 예측은 팀 `experiments/valid_predictions_zoo.csv.gz`의 보정 전 XGBoost 열을 사용한다. 나머지 예측은 `sincon_xgb_forecasts.zip` 및 `sincon_xgb_extra_validation.zip`에서 확인할 수 있다.
5. `sincon_clean_15126_replay.py`에 팀 폴더, 2025년 과거 예측 폴더, 최종 XGBoost, 새 CSV, 보고서 경로를 넘긴다. 재학습한 경우 `--local-stack`, `--conditional-stack`, `--local-historical-abc`로 1단계 출력을 지정한다. 과거 행별 예측도 새로 만들려면 `--validate --validation-archive` 및 `--recent`, `--extra-xgb`, `--june-xgb`, `--april-xgb`를 지정한다. 아홉 구간의 `combined` 열이 후속 모형과 행사 검증에 쓰이는 보정 전 기준이다. 이 열을 `baseline`으로 이름을 바꾸고 `date,station_id,hour,label,origin,baseline` 순으로 CSV에 저장한다.
6. 공연 취소와 송도해변축제 장소 변경의 원본 모델 반응은 `sincon_munhak_cancellation_lgb.py`, `sincon_songdo_beach_location_lgb.py`, `sincon_munhak_cancelled_nn.py`로 재학습한다. 송도 LightGBM 스크립트에는 2025-07-31 예측과 5단계의 CSV를 준다. 신경망은 시드 0·1·2를 각각 실행한다. 송도 시나리오에는 `--songdo-beach`를 넣는다. 재현 경로는 두 `*_responses.zip`에 보관한 보정 전 모델 응답을 사용하며, 96개 행의 수정량을 이 응답에서 재계산한다.
7. `sincon_clean_tabicl_forecast.py`에 팀 폴더, 로컬 노트북이 만든 `external.zip`, 정확한 TabICL 체크포인트, 출력 경로를 전달한다. 최종에는 `--folds final`을 지정한다. 과거에는 `--folds A B C summer 2025-07-31 2025-08-31 2025-09-30 2026-03-31 2026-04-30`을 지정한다. 벡터 XGBoost는 `sincon_archived_intraday_anomalies.vector_inputs`와 `vector_signal`에 각 기준일의 조건별 결합 예측을 넘긴다. 최종 열은 `date,station_id,hour,fold,raw_delta`다. A/B/C는 `sincon_vector_boost.py`의 검증 출력에서 `shape-reference`를 사용한다. 보관된 벡터 입력은 재현 중 해시와 키가 확인된다.
8. `sincon_clean_event_replay.py --stage before_model`로 행사 보정을 만들고 `--protected`로 보호 키를 남긴다. `sincon_clean_model_layers.py`에 그 예측과 보호 키, 5단계의 OOF, 7단계의 TabICL·벡터 입력을 준다. `sincon_clean_event_replay.py --stage after_model`로 검암, 선학, 과학대제전 시간대와 둘레길 축제 보정을 적용한다. 정확 재현 명령이 이 세 단계를 순서대로 실행한다. 1·4단계의 새 모델을 활용하려면 명령에 `--local-stack`, `--conditional-stack`, `--local-historical-abc`, `--xgb`를 지정한다.

로컬에서는 공식 CSV와 보관 신경망을 사용한 원본 스택 재학습, 최종 XGBoost 재학습, TabICL 10개 기준일 원시 예측 생성, 정리본 조립을 실행했다. 팀 MLP 30개 시드 전체를 새 런타임에서 재학습하는 경로와 Kaggle·Colab에서의 전체 재실행은 확인 전이다. 정확 재현에는 검증된 보관 예측을 사용한다.

### 시드와 환경

| 구성 | 시드와 고정 설정 |
|---|---|
| 팀 LightGBM 및 생활이동 LightGBM | 0·1·2, 1,500회, 학습률 0.05, 63 leaves, 최소 리프 100, 로그 승차 MSE |
| 팀 임베딩 MLP | 최종과 A/B/C는 0~29, 30 epoch, 폭 512, dropout 0.15. 2025년 월말 롤링 예측은 시드 0, 30 epoch, 폭 384 |
| 그래프와 혼합 신경망 | 시드 0·1. 그래프 120 epoch, 비음수 혼합 125 epoch 및 6개 성분 |
| 버스 LightGBM과 행사 대응 | 버스 시드 0. 취소·장소 LightGBM 시드 0·1·2, MLP 대응도 0·1·2 |
| 시간별 XGBoost | 시드 0, 2,000회, 학습률 0.05, 깊이 8 |
| 벡터 XGBoost | 시드 19, 650회, 학습률 0.06, 깊이 6, 18출력 트리 |
| TabICL | 시드 2026, CPU, 앙상블 멤버 2. [사용 체크포인트](https://huggingface.co/jingang/TabICL/blob/main/tabicl-regressor-v2-20260212.ckpt) |

| 확인한 실행 | 패키지 버전 |
|---|---|
| 원본 로컬·조건별 스택 매니페스트 | numpy 2.3.5, pandas 2.2.3, scipy 1.17.1, torch 2.9.1, lightgbm 4.6.0 |
| 이번 TabICL 및 정리본 조립 실행 | Python 3.12.15, macOS ARM64, numpy 2.5.4, pandas 3.1.1, scipy 1.17.1, torch 2.15.0, lightgbm 4.6.0, xgboost 3.4.1, scikit-learn 1.9.1, tabicl 2.2.0, catboost 1.2.10, huggingface-hub 2.4.0, einops 0.8.2, psutil 7.2.2 |

검증 로그에 포함된 모델별 데이터 기준일은 최종 2026-06-30, A 2026-02-28, B 2025-12-31, C 2025-10-31, 엄격 여름 2025-06-30이다. 정답 조회는 해당 검증 및 학습 기간의 공식 `train.csv`로 제한한다.


## 2. 3단계의 모델 예측 6개 (`code/3_stack/training_scripts/`)

아래는 3단계 패키지를 만들 때 적은 설명이다. 본문의 `xgb7_inputs/` 는 원 작업 폴더 이름이다.

### 6개 모델 예측을 다시 만드는 방법

이 절은 보관 입력의 생성 방법이다. 기본 노트북은 아래 학습을 수행하지 않는다. 보관 npy에는 ID가 없으므로 순서는 사용자 제공 설명과 원 `build_xgb_delta.py`의 test 피처 키 확인에 근거한다. 이 패키지에서는 각 배열의 길이·유한값·SHA와 공식 test 순서를 고정하지만, 배열만 보고 원래 순서를 독립적으로 증명할 수는 없다.

`training_scripts/xgb_test.py`, `run3.py`, `run2.py`, `prep.py`, `prep_test.py`는 `xgb7_inputs/scripts/`에서 수정 없이 복사했다. `set2_build_l1.py`와 `build_xgb_delta.py`는 하드코딩 경로·작업 디렉터리를 제거한 실행 진입점이며 실제 계산은 노트북과 같은 함수를 쓴다. 원본의 상수·연산 순서는 바꾸지 않았다.

원 피처 목록은 `team_features/ensemble.py`의 `FEATURES`, 전처리는 `improved.enrich`, 달력은 `make_calendar.build_calendar`다. 전처리에는 달력·역·노선·경기·관중·행사·학교·날씨 특징을 사용한다. 승하차 목표는 공식 train의 2026-06-30까지뿐이다. `ensemble.py`는 피처 목록과 신경망 설정을 읽기 위해 import된다. 파일에 남아 있는 별도 제출 경로의 Public 기반 수준 보정 상수는 아래 스크립트에서 사용하지 않는다.

패키지를 읽기 전용으로 둔다면 `training_scripts/` 전체를 별도 쓰기 가능한 `training_work/`로 복사한다. 이 폴더에서 필요한 학습 라이브러리가 미리 설치된 환경으로 다음 순서로 실행한다. `requirements-training.txt`는 선택적 재학습용 버전 예시이며, 보관 학습 당시 전체 환경의 검증된 lockfile은 아니다. XGBoost 3.4.1은 사용자 제공 버전이다.

```sh
cd training_work
python -B prep.py team_features data.pkl
python -B prep_test.py team_features test.pkl
mkdir -p out3
python -B xgb_test.py
python -B run3.py data.pkl test.pkl base mps out3 0,1,2,3
```

Linux에서는 마지막 명령의 `mps`를 `cpu` 또는 `cuda`로 바꿀 수 있다. 그러면 새 예측이 만들어지지만 보관값과 같다고 볼 수 없다. 보관 파일을 자동으로 대체하지 않는다.

| 모델 | 원 설정 |
|---|---|
| XGBoost | 버전 3.4.1, `tree_method=hist`, 깊이 8, 학습률 0.05, 800그루, `subsample=0.8`, `colsample_bytree=0.8`, `enable_categorical=True`, `n_jobs=8`, seed 0. 목표는 float64 `log1p(boardings)`. |
| LightGBM | 600그루, regression, 학습률 0.05, 잎 63, 잎 최소 50행, 열·행 추출 0.8, `bagging_freq=1`, 8스레드, seed 0. 목표는 같은 log 승차. |
| 팀 base MLP | 시드 0~3, Mac GPU MPS 학습 보관본. 30 epoch, 폭 512·512·256, SiLU, dropout 0.15, AdamW 학습률 0.002·weight decay 1e-5, batch 2048, OneCycleLR `pct_start=0.1`. 학습 구간으로 수치형 절단·표준화와 log 목표 표준화를 적합한다. 임베딩 정의는 `run2.py`, 전체 적합은 `run3.py`. |

앙상블은 `0.5 × lgb + 0.5 × mean(base_s0..3)`이다. MLP 보관 배열은 float32이며 평균을 float64로 바꾸지 않는다. 최종 차이도 원 NumPy 연산 순서를 그대로 따른다. 제외 역은 `3107, 3108, 3109, 3127, 3139, 3209, 3754`다. 4개 달력 표시가 모두 0이고 `events_frozen_v3` 및 `protected_event_keys`의 역·날짜가 아닌 13,662행에만 0.1배 차이를 더한다. 원본과 같이 최종 log 예측을 0 이상으로 제한한다.

보관 MLP는 다른 하드웨어에서 수치가 달라지며, 이 작업에서는 6개 모델을 재학습하지 않았다. 피처 생성 소스와 입력도 함께 보관했지만 그것들로 해당 npy가 생성되었는지 학습을 다시 수행해 확인하지는 않았다.


## 3. 4단계의 타 노선 자료와 XGBoost 예측 2개 (`code/4_rail/`)

| 순서 | 스크립트 | 하는 일 |
|---|---|---|
| 1 | `fetch/fetch_all.py` | OA-12914 월별 파일을 받아 7호선과 환승역을 빼고 `cache/rail_all_allowed.csv.gz` 를 만든다. 원본은 지운다. 제외 목록은 `cache/ban_list.json`, 파일 순번은 `fetch/seq_map.json` |
| 2 | `fetch/fetch.py` | 같은 자료에서 경인선·수인선 허용 25개 역만 남겨 `training/rail_allowed.csv.gz` 를 만든다 |
| 3 | `training/prep.py`, `prep_test.py` | 3단계와 같은 팀 피처로 학습·시험 프레임(`data.pkl`, `test.pkl`)을 만든다 |
| 4 | `training/nb9.py` | 인접 노선 날짜 피처 7개(경인선·수인선·전체의 일 승차 합 log, 같은 요일 직전 4주 중앙값 대비 비율, 하차 비율)를 만들고 과거 기준일별 XGBoost 예측을 만든다 |
| 5 | `training/xgb_test.py`, `nb_test.py` | 피처 없이 / 피처를 넣어 전체 학습 후 시험 기간 예측 → `predictions/xgb_test.npy`, `xgbnb_test.npy` |

- XGBoost 3.4.1, `tree_method=hist`, 깊이 8, 학습률 0.05, 800그루, seed 0 (3단계의 XGBoost와 같은 설정).
- 스크립트는 원 작업 폴더의 상대 경로를 쓴다. 이 폴더 구조에 맞게 고치지 않았다. 중간 파일(`data.pkl`, `test.pkl`, `nb_features.pkl`)은 보관하지 않았다. 다시 학습해 보관본과 대조하지 않았다.
- `validation/` 은 계수를 정한 과거 검증 스크립트와 결과(JSON)다. 입력인 기준일별 과거 예측 묶음은 3단계의 `cache/downstream_oof_nested_main.csv.gz` 와 기준일별 XGBoost 예측인데, 뒤의 것은 보관하지 않았다.

## 4. 순방향(중첩) 과거 예측 (`code/3_stack/oof_generation/`)

`cache/downstream_oof_nested_main.csv.gz` 는 9개 기준일마다 그날까지의 정답만으로 1단계의 층 채택을 다시 정해 만든 과거 예측이다.

- `night11_reproduce_nested.py`, `night11_reproduce_all.sh`: 1단계의 기준일별 보관 예측에서 중첩 모델층을 다시 실행한다.
- `night13_amp_final_layers.py`, `night13_extract_final_layers.py`, `night13_common.py`: 2단계의 보정 6개를 기준일별로 적용해 최종 과거 예측을 뽑는다.
- 스크립트는 원 작업 폴더의 경로를 가정한다. 제출 준비 중에는 다시 실행하지 않았다.
