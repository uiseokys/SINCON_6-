# 데이터 출처, 생성 경로, 모델·중간 산출물

## 1. 제공 데이터에서 최종 예측까지

```
train.csv, test.csv, metaData.csv (제공)  +  외부 자료 (2절)
        │
        ▼  [1] code/1_base_models/SINCON_final.ipynb
        │      full: 모델 학습 → 모델별 예측   /  assemble: 보관한 모델별 예측을 읽음
        │      → 결합(스택) → 하위 보정 → 행사 보정
        ▼
   submission_clean_refined.csv
        │
        ├──▶ [2b] code/2b_monday/SINCON_monday.ipynb ─▶ submission_monday_main.csv ────────┐
        ▼  [2] code/2_corrections/SINCON_final_step2_layers.ipynb (보정 6개)               │
   submission_night_v4_main.csv                                                             │
        │                                                                                   │
        ├──▶ [2c] code/2c_cup_walk/model/build_v4_cup_walk.py                              │
        │         ─▶ submission_night_v4_main_cup_walk.csv ─────────────────────────────────┤
        ▼  [3] code/3_stack/SINCON_final_stack.ipynb (7개 단계)  ◀──────────────────────────┘
        │      + 순방향 과거 예측(cache/), 모델 예측 6개(predictions/)
        ▼
   submission_xgb7_w10_on_curvature.csv
        │
        ▼  [4] code/4_rail/SINCON_rail_factors.ipynb
        │      + 타 노선 일별 승차(cache/rail_all_allowed.csv.gz), XGBoost 예측 2개(predictions/)
        ▼
   submission_rail_factors_k30_hour_on_xgb7.csv   ← 최종 예측 (predictions/submission_final_main.csv)
```

- [3]은 [2b], [2c]의 출력을 패키지 안의 보관본(`code/3_stack/inputs/`)으로 읽는다. 전체 실행 노트북(`SINCON_full_pipeline.ipynb`)은 두 출력을 다시 만들어 보관본과 대조한다.
- 입력 파일의 해시: `code/1_base_models/payload_manifest.json`, `code/2_corrections/delivery_manifest.json`, `code/3_stack/SOURCE_MANIFEST.json`.

## 2. 데이터 출처 작성표

### 대회 제공 데이터 (변형 없음)

| 파일 | 위치 |
|---|---|
| `train.csv`, `test.csv`, `metaData.csv` | `code/1_base_models/payload/team/data/` |
| 같은 파일의 사본 | `code/3_stack/data/`, `code/3_stack/training_scripts/team_features/data/`, `code/4_rail/data/` (각 단계가 단독으로 실행되도록 넣음) |

`sample_submission.csv` 는 쓰지 않았다.

### 외부 데이터

정본은 `sources/EXTERNAL_DATA.md`(출처 링크·조회일·이용 조건·7월 이후 정보 포함 여부), `sources/EXTERNAL_SOURCES_EVENTS.md`(행사별 URL), `sources/STAGE2_SOURCES.md`(2단계 입력)다. 아래는 요약이다.

| 자료 | 출처 | 조회일 | 이용 조건 | 예측 기간(7~10월) 값 | 단계 |
|---|---|---|---|---|---|
| 공휴일·선거일·연휴 | 공휴일에 관한 법률, 천문연 월력요항 | 2026-11-11 | 공개 날짜 | 예정 날짜 | 1~4 |
| NEIS 학사일정 | data.go.kr 15137088 | 2026-11-11 | 제한 없음 | 포함 | 1 |
| 대학 방학 | 인천대 공문, 고등교육법 시행령 | 2026-11-11 | 날짜 사실 | 포함 | 1 |
| Open-Meteo 시간별 날씨 | open-meteo.com | 2026-11-11 | CC BY 4.0 | **실제 날씨** | 1 |
| 서울시·KT 생활이동 OA-22655 (비지하철 수단) | data.seoul.go.kr | 최초 기록 없음, 재확인 2026-11-12 | 공공누리 1유형 | **실제 이동량** | 1 |
| 인천 버스 정류장별 월간 승하차·좌표 | data.go.kr 15048264, 15067528 | 최초 기록 없음, 재확인 2026-11-12 | 제한 없음 | **실제 버스 이용량** | 1 |
| 인천교통공사 역 주소·출입구 좌표 | data.go.kr 15043811, 15166820 | 최초 기록 없음, 재확인 2026-11-12 | 제한 없음 | 정적 정보 | 1 |
| K-apt 단지 승인·입주월·세대수·좌표 | k-apt.go.kr, data.go.kr 15073271 | 최초 기록 없음 | 일부 미확인 | 입주월 | 1 |
| 문체부 지역축제 계획·표준데이터 | mcst.go.kr, data.go.kr 15013104 | 최초 기록 없음 | 유형 미확인 | 일정 | 1 |
| KBO 문학 경기·결과·관중·종료 시각 | koreabaseball.com | 2026-11-11 | **허용 미확인**. CC BY-NC-SA 위키와 대조 | **실제 경기 정보** | 1 |
| K리그 부천 경기·킥오프·관중 | kleague.com | 최초 기록 없음 | 미확인. 위키와 대조 | **실제 경기 정보** | 1 |
| 팀 수기 공연·행사, 취소·장소 공지 | 주최 측 페이지 | 2026-11-11~12 | 일정 사실. 개별 조건 미확인 | 일정·취소 사실 | 1~3 |
| 동춘 대형마트 정기 휴무 | 연수구 운영표 | 2026-11-12 | 미확인 | 정기 규칙 | 1 |
| PLAVE 공연 예매 수량 | 팀원 제공(KOPIS라고 전달받음) | 기록 없음 | 미확인 | 2행에 사용 | 2 |
| 코리아컵·걷기대회 일정 | `code/2c_cup_walk/graph_clean/bucheon_cup_facts.csv`, `bottleneck/incheon_walk.csv` 에 행별 출처·조건·조회일 | 파일에 기록 | 파일에 기록 | 일정 | 3 |
| TabICL 2.2.0, 체크포인트 `20260212` | PyPI, Hugging Face `jingang/TabICL` | 2026-11-12 확인 | BSD-3-Clause (`code/1_base_models/licenses/`) | 사전학습 모델 | 1 |
| **타 노선 일별 승차 OA-12914** | data.seoul.go.kr/dataList/OA-12914 | **2026-11-13** | 공공누리 1유형 (서울특별시·코레일·공항철도) | **실제 지하철 승차(타 노선)** | **4** |

타 노선 자료의 범위·제외 목록·규칙 관련 사실은 `DISCLOSURE.md` 2절. 데이터셋 페이지 사본은 `code/4_rail/fetch/OA-12914_dataset_page_20261113.html`.

### 파생 데이터·캐시

| 파일 | 위치 | 내용 | 만든 방법 |
|---|---|---|---|
| 기준일별 과거 예측 | `code/1_base_models/payload/` 의 ZIP들 | 1단계 모델의 기준일별 예측 | `RETRAINING.md` 1절 |
| `downstream_oof_nested_main.csv.gz`, `hourbias_history_oof.csv.gz` | `code/3_stack/cache/` | 주후보의 순방향(중첩) 과거 예측과 실측 log. 뒤의 것은 앞의 것에서 날짜만 잘라 낸 것 | `code/3_stack/oof_generation/` (`RETRAINING.md` 4절) |
| 달력·행사·보호 행 | 각 단계의 `cache/` | 공휴일류 표시, 행사 역·날짜 | `make_calendar.py`, 행사 목록(출처 열 포함) |
| `rail_all_allowed.csv.gz`, `ban_list.json` | `code/4_rail/cache/` | 허용한 타 노선 561개 역의 일별 승차, 제외 이름 목록 | `code/4_rail/fetch/fetch_all.py` |
| `rail_allowed.csv.gz` | `code/4_rail/training/` | 경인선·수인선 허용 25개 역의 일별 승차 | `code/4_rail/fetch/fetch.py` |

## 3. 모델·중간 산출물 작성표

| 단계 | 산출물 | 위치 | 다시 만드는 방법 | 재생성 확인 |
|---|---|---|---|---|
| 1 | 팀 LightGBM(시드 3)·MLP(시드 30) 예측 | `code/1_base_models/payload/team/ensemble_parts.csv`, `valid_predictions_ABC.csv.gz` | 1단계 노트북 `full`, 단계 `native` | 전체 재학습 1회. MLP가 보관본과 다름(최대 log 0.097, RMS 0.005) |
| 1 | 스택 모델 출력과 과거 예측 | `payload/work/sincon_local_validation.csv.gz`, `sincon_local_repro.zip`, `sincon_source_reweight_portable.zip`, `sincon_rolling_conditional_cache.zip`, `sincon_summer_replay_validation.csv.gz`, `sincon_target_season_validation.zip` | `full`, 단계 `stacks`, `rolling` | 전체 재학습 1회. 2025년 벡터 입력의 일부 선행 예측은 `sincon_rolling_conditional_cache.zip` 보관본을 다시 씀 |
| 1 | 시간별 XGBoost, 벡터 XGBoost | `payload/work/sincon_xgb_forecasts.zip`, `sincon_xgb_extra_validation.zip`, `sincon_vector_shape_final_raw.csv.gz`, `sincon_vector_boost_validation.csv.gz` | `full`, 단계 `hourly_xgb`, `vector` | 위와 같음 |
| 1 | TabICL 원시 예측, clean OOF, 행사 응답 | `payload/work/sincon_clean_refined_cache.zip`, `sincon_munhak_cancellation_responses.zip`, `sincon_songdo_beach_location_responses.zip` | `full`, 단계 `tabicl`, `clean_oof`, `event_responses` | 위와 같음 |
| 1 | **사용한 초기 모델**: TabICL 회귀 체크포인트 | `payload/models/tabicl-regressor-v2-20260212.ckpt` (109MB, SHA-256 `0db9cb538f114e79…`) | Hugging Face `jingang/TabICL` 에서 내려받음 | 해시 고정 |
| 2 | 층 계산용 캐시 | `code/2_corrections/cache/` | 과거 예측 요약과 행사 일정에서 정리 | 학습하는 모델 없음 |
| 3 | 순방향 과거 예측 | `code/3_stack/cache/` | `oof_generation/` | 제출 준비 중에는 다시 실행하지 않음 |
| 3 | 월요일 후보, 코리아컵·걷기 후보 | `code/3_stack/inputs/` | `code/2b_monday/`, `code/2c_cup_walk/` | **이번 실행에서 다시 만들어 대조: 최대 log 차이 0, 8.9e-16** |
| 3 | 단계 7 모델 예측 6개 (XGBoost, LightGBM, MLP 4시드) | `code/3_stack/predictions/*.npy` | `training_scripts/` (`RETRAINING.md` 2절) | 다시 학습해 대조하지 않음. MLP는 Apple MPS에서 학습한 보관본 |
| 4 | 타 노선 일별 승차 | `code/4_rail/cache/rail_all_allowed.csv.gz` | `fetch/fetch_all.py` | 다시 내려받아 대조하지 않음 |
| 4 | XGBoost 예측 2개 (인접 노선 피처 유무) | `code/4_rail/predictions/*.npy` | `training/` (`RETRAINING.md` 3절) | 다시 학습해 대조하지 않음 |
| 4 | 주성분·Ridge 계수 | 보관하지 않음 | 노트북이 실행할 때마다 계산 | 결정적 계산 |

- 1단계 전체 재학습 결과와 보관본의 차이: 평균 절대 log 0.0012, RMS 0.0027, 최대 0.153, 99.85%의 행이 0.01 이내. 검증 점수는 A 0.149912 / 0.149904, B 0.164563 / 0.164576, C 0.161220 / 0.161175 (재학습 / 보관본).
- **해당 없음**: 임베딩 파일, 별도 저장한 최종 가중치 파일(신경망·트리 모델의 가중치는 저장하지 않고 예측만 보관했다), 생성형 모델, 대상 역의 2026년 7월 이후 승하차를 담은 파일.
