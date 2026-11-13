# 방법 설명서 — 팀 핫식스

과제: 인천 지하철 71개 역의 2026-07-01~10-31 시간대별(6~23시) 승차 인원 예측, 157,194행, 평가 지표 RMSLE. 학습 자료는 `train.csv`(2025-01-01~2026-06-30).

최종 예측(`submission_final_main.csv`)은 네 단계를 차례로 쌓아 만든다. 각 단계는 앞 단계의 예측 CSV를 입력으로 받는다.

| 단계 | 패키지 | 하는 일 | 출력 (Public) |
|---|---|---|---|
| 1 | `code/1_base_models/` | 기본 모델들의 예측을 결합하고 하위 보정을 적용 | `submission_clean_refined.csv` (0.148046) |
| 2 | `code/2_corrections/` | 검증한 보정 6개 | `submission_night_v4_main.csv` (0.146049) |
| 3 | `code/3_stack/` | 보정 7개 단계 | `submission_xgb7_w10_on_curvature.csv` (0.144715) |
| 4 | `code/4_rail/` | 다른 노선 자료를 쓰는 세 층(주성분 30개) | `submission_rail_factors_k30_hour_on_xgb7.csv` (0.143043) → **최종 예측** = `submission_final_main.csv` |

Public 점수는 팀장이 Kaggle에서 확인해 전달한 값이다.

## 1. 목표 변수와 공통 사항

- 목표는 `log1p(boardings)`. 모든 보정은 log 공간에서 더하고 `expm1` 로 되돌린다. 음수는 0으로 자른다.
- 대상 역의 승하차 실측은 `train.csv`(2026-06-30까지)만 쓴다.
- 행사·명절처럼 따로 보정한 역·날짜는 "보호 행"으로 두고 뒤 단계의 일반 보정이 건드리지 않게 했다(`protected_event_keys.csv`, `events_frozen_v3.csv`, `team_event_station_days.csv`).

## 2. 검증 분할

- 시간 순서 분할만 쓴다. 기준일까지의 자료로 맞추고 그 뒤 약 4개월을 맞히는 방식이며, 기준일은 9개다: 2025-06-30, 07-31, 08-31, 09-30, 10-31(C), 12-31(B), 2026-02-28(A), 03-31, 04-30. 뒤쪽 두 기준일은 평가 기간이 2026-06-30에서 끝나 더 짧다.
- 1단계 모델 비교는 주로 A/B/C 세 구간으로 했다.
- 2~4단계의 보정은 **순방향(중첩) 과거 예측** 위에서 쟀다. 각 기준일에서 그날까지의 정답만으로 1단계의 층 채택까지 다시 정한 예측이다(`code/3_stack/cache/downstream_oof_nested_main.csv.gz`). 원시 모델을 기준일마다 다시 학습한 것은 아니고, 보관한 기준일별 원시 예측에서 층 채택과 결합을 다시 한 것이다.
- 기준일들의 평가 기간은 서로 겹친다. "9개 중 8개 개선"은 독립된 9번의 시험이 아니다.

## 3. 1단계 — 기본 모델과 결합 (`code/1_base_models/`)

구성과 시드의 정본은 `RETRAINING.md` 1절의 "시드와 환경" 표다. 요약:

| 구성 | 설정 |
|---|---|
| 팀 LightGBM | 시드 0·1·2, 1,500회, 학습률 0.05, 63 leaves, 최소 리프 100, log 승차 MSE |
| 팀 임베딩 MLP | 시드 0~29, 30 epoch, 폭 512, dropout 0.15 |
| 생활이동 LightGBM, 버스 LightGBM | 시드 0·1·2 / 0 |
| 그래프 신경망, 수요 혼합 신경망 | 시드 0·1, 120 epoch / 125 epoch(6개 성분) |
| 가법 시간대 모형, 경기 모형, 축제·주거 보정 | 결정적 계산 |
| 시간별 XGBoost | 시드 0, 2,000회, 학습률 0.05, 깊이 8 |
| 벡터 XGBoost(하루 18시간을 한 번에) | 시드 19, 650회, 학습률 0.06, 깊이 6 |
| TabICL 2.2.0 (공개 사전학습 모델, 체크포인트 `tabicl-regressor-v2-20260212.ckpt`) | 시드 2026, CPU, 앙상블 멤버 2 |
| 행사 응답(공연 취소·장소 변경) | LightGBM 시드 0·1·2, MLP 시드 0·1·2 |

- **주요 변수**: 팀 기본 모델(LightGBM·MLP)의 변수는 `ensemble.py` 의 `FEATURES` = 달력(역·시간·요일·월·공휴일류) + 경기 + 관중 + 행사 + 학사 일정 + 날씨다. 그 위의 모델들은 역 주변 버스·생활이동 집계, 역 간 그래프, 아파트 입주 정보, 축제 일정을 더 쓴다. 변수 생성 코드는 `code/1_base_models/payload/team/` (`baseline.py`, `improved.py`, `external_features.py`, `make_calendar.py`, `ensemble.py`)와 `payload/work/` 의 스크립트.
- **결합 비중**: 모델 결합(스택)의 가중과 하위 보정의 채택은 2026-06-30 이전 검증 구간의 오차로 정했다. 층별 채택 결과와 수치는 `code/1_base_models/payload/work/` 의 `*_results.json` 과 `code/1_base_models/docs/` 의 JSON에 있다. 이 문서에 개별 가중치를 옮겨 적지 않은 것은 값의 정본이 그 파일들이기 때문이다.
- **Public 유래 값 제거**: 원본 스택에 있던 두 값(전체 수준 +0.009024, 역 패턴 비율 0.62)은 이 경로에서 뺐다(`DISCLOSURE.md` 1절).

## 4. 2단계 — 보정 6개 (`code/2_corrections/`)

| 순서 | 보정 | 바뀐 행 | 값을 정한 방법 | 과거 검증 |
|---|---|---|---|---|
| 1 | 타임캡슐 슈퍼콘서트 (7/11, 송도달빛축제공원 21~23시) | 3 | 2025년 같은 공연의 실제 초과 승차의 75% | 같은 방식을 2025 펜타포트·맥주축제에 적용: 해당 행 RMSLE 1.70 → 1.41 |
| 2 | PLAVE 콘서트 (9/12~13, 문학경기장 21~22시) | 4 | 과거 주경기장 공연 수준 쪽으로 35% | 과거 공연 14행 중 13행 개선 (2.08 → 1.66) |
| 3 | 역별 최근 오차 보정 | 전체 | 기준일 직전 60일의 과거 예측 오차를 역별로 평균해 0.25배 | 8개 기준일 중 7개 개선 (−0.0002~−0.0043), 1개 악화 (+0.0016) |
| 4 | 금요일 공휴일 (7/17, 10/9) | 2,016 | 과거 금요일 공휴일의 역별 차이, log ±0.05 이내, 0.2배 | 발동한 5개 기준일 모두 개선 |
| 5 | PLAVE 22시 | 2 | 과거 공연의 22시 초과 승차 ÷ 관객 수(0.173) × 예매 수량 | 과거 공연 22시 5행 RMSLE 2.57 → 0.59~0.61 |
| 6 | 부평삼거리역 추석 전날·당일 (9/24~25) | 36 | 과거 명절 당일의 평소 일요일 대비 배율 | 발동한 6개 기준일 모두 개선 (−0.0001~−0.0006) |

근거 문서: `records/development_history.zip` 의 `dev_logs/team/FINAL_REPORT.md` 4절, `removed_from_packages/final_delivery_v2/VALIDATION.md`.

## 5. 3단계 — 보정 7개 단계 (`code/3_stack/`)

| 단계 | 내용 | 계수 | 과거 검증 (전체 RMSLE 변화) |
|---|---|---|---|
| 1 K1 | 일반일의 역×요일×시간대 오차 보정 | 최근 84일, 수축 n/(n+12), 0.25배, 최소 6일 | 7개 기준일 중 5개 개선, 평균 −0.00013 |
| 2 K3 | K1 + 월요일 대체공휴일(8/17, 10/5) 2,016행 + 코리아컵 4행·걷기대회 7행 | 월요일: 금요일 공휴일과 같은 식(0.2배, ±0.05). 걷기대회: 2025년 초과 승차의 50% | 월요일 4/4 개선(−0.000006~−0.000033). 행사는 과거 사례 1건씩 |
| 3 | 검단호수공원역(3107) 수준, PLAVE 공연 중 4행 | 수준 log +0.0748 | 3107: 발동한 3개 기준일 모두 개선(−0.0006~−0.0010). PLAVE: 과거 12행 0.576 → 0.509 |
| 4 | 계절 재적합 | K1 적용 후 남은 과거 오차로 계절 보정을 다시 계산 | 8개 중 6개 개선, 평균 약 −0.00005 |
| 5 L1 | K1 규칙을 제외했던 7개 역(3107, 3108, 3109, 3127, 3139, 3209, 3754)으로 확장 | K1과 같음 | 8개 중 1개 악화, 평균 −0.00004 |
| 6 | 시간대 고주파 패턴(curvature) | 실측 프로필의 고주파 성분 15% | 9개 중 6개 개선, 평균 −0.00006 |
| 7 XGB7 | 7개 역 일반일에 `0.1 × (XGBoost − 대체 앙상블)` | 대체 앙상블 = 0.5×LightGBM + 0.5×MLP 4시드 평균 | 사전 기록은 개발 기록 묶음의 `CANDIDATE_SET2_PREREGISTRATION.md` |

- 단계 1~6은 pandas·numpy 계산이다. 단계 7은 보관한 모델 예측 6개(`predictions/*.npy`)를 읽는다. 모델 설정과 재학습 명령은 `RETRAINING.md` 2절.
- 층별 점검, 사전 기록, 단계별 근거 문서는 `records/development_history.zip` 에 있다: `dev_logs/team/STACK_AUDIT_1113.md`, `CANDIDATE_SET_PREREGISTRATION.md`, `CANDIDATE_SET2_PREREGISTRATION.md`, `removed_from_packages/final_stack_delivery/evidence/`.

## 6. 4단계 — 다른 노선 자료를 쓰는 층 (`code/4_rail/`)

입력: 7호선과 환승역을 뺀 다른 노선 544개 역의 일별 승차(`cache/rail_all_allowed.csv.gz`). 규칙 관련 사실은 `DISCLOSURE.md` 2절.

| 층 | 방법 | 계수 |
|---|---|---|
| 변화 옮기기 | 타 노선 역의 `log(1+일 승차)`를 "같은 요일 직전 8주 중앙값 대비 증감"으로 바꿔 2026-06-30 이전 날짜로 주성분을 구한다. 대상 역마다 자기 증감을 주성분에 Ridge로 적합한다. 시험 기간 날짜마다 "주성분으로 본 예상 변화 − 우리 예측의 변화"를 일부 더한다 | 주성분 30개, Ridge α=5, 일반일 0.2·공휴일류 0.3, 행당 ±0.15 |
| 공휴일류 수준 | 인접 노선(경인선·수인선) 날짜 피처 7개를 넣은 XGBoost와 넣지 않은 XGBoost의 차이를 역·날짜 평균으로 만들어 공휴일류 날짜에 더한다 | 0.5배, ±0.15 |
| 시간대별 반응 | 역·시간대마다 증감을 주성분에 Ridge로 적합해 "예상 형태 변화 − 우리 예측의 형태 변화"를 더한다(역·날짜 평균은 유지) | 주성분 12개, α=5, 0.1배, ±0.1 |

- 보호 역·날짜는 세 층 모두 건드리지 않는다: 행사 목록(`events_frozen_v3.csv`, `protected_event_keys.csv`, `team_event_station_days.csv`)과 기존 보정이 있는 (2026-09-24, 3122), (2026-09-25, 3122), (2026-07-11, 3139), (2026-07-29, 3754), (2026-10-17, 3226).
- **계수 결정**: 채택 기준은 "현재 구성 대비 9개 기준일 중 악화 1개 이하이고 평균이 개선, 그중 평균이 가장 좋은 하나"로 미리 적었다. 주성분 30개·0.2/0.3 구성은 현재 구성(5개·0.1/0.2) 대비 평균 −0.00083, 악화 1개(2026-02-28 기준일 +0.00044)였다. 주성분 12개·0.15/0.3 구성은 평균 −0.00049, 악화 0개였다(악화가 없는 구성 중 평균이 가장 좋음). 두 구성 모두 과거 자료로만 정한 완성 후보로 만들어 제출했다. 시간대 층은 9개 중 8개 개선, 평균 −0.00021(2025-08-31 기준일 +0.00068)이었다.
- 검증 스크립트와 결과는 `code/4_rail/validation/` (`factor_scan.py`·`factor_scan2.py` 가 주성분 수와 가중, `hour_factor_k1.py`·`hour_factor_k30.py` 가 시간대 층, `nb_val2.py` 가 공휴일류 수준, `lunar_check.py` 가 보호 역·날짜). 전 과정(시도, 탈락, 시각)은 `records/development_history.zip` 의 `dev_logs/team/RAIL_PREREGISTRATION.md`.
- 알려진 약점: 서울에만 생긴 변동(예: 2026-01-13~14 서울 시내버스 파업)을 대상 역에도 옮긴다(개발 기록의 `DISRUPTION_TEST_1113.md`).

## 7. 제출한 후보와 선택

값은 과거 자료로 정하고, 완성한 후보를 제출해 Public이 좋은 것을 골랐다.

| 파일 | Public |
|---|---:|
| `submission_clean_refined.csv` (1단계) | 0.148046 |
| `submission_night_v4_main.csv` (2단계) | 0.146049 |
| `submission_amp15_main.csv` | 0.146197 |
| `submission_monday_main.csv` | 0.146049 |
| `submission_night_v5_events_hedged15.csv` | 0.146236 |
| `submission_k1_hourbias.csv` | 0.145606 |
| `submission_k2_holiday_events.csv` | 0.146049 |
| `submission_k3_hourbias_holiday_events.csv` | 0.145606 |
| `submission_k3_geomdan_plave_seasonal_refit.csv` | 0.145104 |
| `submission_set2_l1_on_refit.csv` | 0.144892 |
| `submission_l1_hourly_curvature.csv` | 0.144804 |
| `submission_xgb7_w10_on_curvature.csv` (3단계 출력, 타 노선 자료 미사용) | 0.144715 |
| `submission_xgbbeta_v2_on_curvature.csv` | 0.144705 |
| `submission_rail_shift_on_xgb7.csv` (인접 25개 역 평균) | 0.144427 |
| `submission_rail_final_on_xgb7.csv` | 0.144280 |
| `submission_rail_factors_on_xgb7.csv` (주성분 5개) | 0.143777 |
| `submission_rail_factors_k12_on_xgb7.csv` (주성분 12개) | 0.143545 |
| `submission_rail_factors_k30_on_xgb7.csv` | 0.143123 |
| `submission_rail_factors_k30_hour_on_xgb7.csv` | **0.143043** (최종 예측) |

- 이 표는 작업 기록에 남은 것만 적었다. 2026-11-12 이전의 제출과 Public 유래 값이 든 제출은 `DISCLOSURE.md` 1절.
- `submission_xgbbeta_v2_on_curvature.csv` 는 Public이 조금 더 좋았지만 재현 노트북을 만들지 않아 최종에서 뺐다.
- 검증했지만 넣지 않은 시도(신경망 변형, CatBoost·ExtraTrees·GRU, Poisson 회귀, 수준 모델, 월별 시간대 분포 옮기기, 요일 구분 계수 등)는 개발 기록 묶음의 `FINAL_REPORT.md` 7절, `CANDIDATE_SET2_PREREGISTRATION.md`, `RAIL_PREREGISTRATION.md`.
