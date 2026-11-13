# 데이터 파일 점검 보고서 · 파일 목록

작성: 데이터 분석·전처리 담당 / 점검일 2026-11-11
환경: Python 3.14.3, pandas 3.1.1, numpy 2.5.4 (Kaggle 기본 pandas 2.x에서는 아직 실행 확인 전)

---

## 0. 먼저 확인할 것 ⚠️

**폴더에 있는 `sample_submission.csv`는 117,576행으로, 9/30까지만 들어 있습니다.**
- 규칙에 따르면 제출 파일은 157,194행이어야 합니다. 빠진 부분은 10월 전체(39,618행, `J117576`~`J157193`)입니다.
- 앞쪽 117,576행의 ID와 순서는 `test.csv`와 정확히 같습니다. 다운로드가 중간에 끊겼거나 주최측 파일 자체가 잘린 것으로 보입니다.
- **조치:** Kaggle Data 탭에서 다시 받아 행 수를 확인해야 합니다. 제출 파일은 반드시 **`test.csv`의 `example_id`(157,194행)를 기준으로** 만들어야 합니다. `features.py`의 final 출력은 이미 test.csv 기준입니다.
- **확인 결과 (2026-11-11):** Kaggle 원본도 117,576행(7~9월분)이었습니다. 그래서 test.csv 기준 157,194행 파일(`submissions/baseline_157194.csv`)을 시험 제출했고, **정상 채점됐습니다(Public RMSLE 0.28264).** 따라서 정답 파일은 157,194행이고, 10월까지 평가됩니다. sample_submission만 잘린 파일이므로 쓰지 말고 test.csv의 ID를 기준으로 제출합니다.

---

## 1. 점검 결과 요약

| 대상 | 점검 항목 | 결과 |
|---|---|---|
| 대회 원본 | train 중복·결측·음수 없음, 행 수 = metaData 관측범위 × 18 (688,176) | PASS |
| | train 역별 첫날·마지막날 = metaData | PASS |
| | test 71역 × 18시간 × 123일 = 157,194, 중복 없음, ID 연번 J000000~J157193 | PASS |
| | train·test 기간 겹침 없음 | PASS |
| | sample_submission = test | **FAIL (위 0번)** |
| 달력 | 669일 연속, 요일 계산, 휴무·연휴 블록 논리 일관성 | PASS |
| | **공휴일 40일이 독립 라이브러리(`holidays` 0.108, KR)와 완전히 일치**. 노동절·제헌절·선거일·대체공휴일 포함 | PASS |
| | 평일 급감일 중 달력으로 설명 안 되는 날 0건 (역 단위로 봐도 0건) | PASS |
| 학교 일정 | 비율 0~1, 결측 없음, 전체 비율 = 초·중·고 가중평균 | PASS |
| | 원본 3개교 방학 시작·종료일을 수기로 대조 | PASS |
| | 방학식·개학식만 등록한 학교 11곳이 방학 0일로 잡히던 문제 | **수정 완료** (0일 학교 11곳 → 6곳) |
| 피처 | 세 기준일 모두 키 중복 없음, 기간·행 수, horizon 1~123, 무한대 없음 | PASS |
| | 정답 컬럼 = train 원본 | PASS |
| | **무작위 50행 × 9개 피처를 별도 코드로 다시 계산해 전부 일치** | PASS |
| | 누출 테스트: 기준일 이후 실측을 999999로 바꿔도 피처가 동일 | PASS |
| 재현성 | 빈 폴더에 코드와 원본만 복사해 다시 실행 → 산출 CSV 5개의 MD5 완전 일치 | PASS |

### 데이터에서 새로 발견한 것 (다음 작업 후보)
- **경기장역에서 행사 때 승차가 크게 튑니다.** 평소의 2.5배를 넘는 역-일 기록은 문학경기장(3127) 131건, 부천종합운동장(3754) 20건, 인천대공원(3226) 11건입니다.
  - 문학경기장은 야구 시즌(3~10월) 밤 21~22시에 급증합니다.
  - KBO 정규시즌 일정은 시즌 전에 발표되므로 규칙상 쓸 수 있는 외부 자료입니다. 단, 출처와 발표일 기록이 필요합니다.
- 승차 0명 행은 76건뿐이고, 절반이 개통 초기 검단호수공원(3107)입니다.

### 알려진 한계
- **valid_main(2025-06-30 기준)에서는 전년 대비·작년 같은 기간 피처가 전부 비어 있습니다.** 2024년 자료가 없기 때문입니다.
- valid_main에서는 신규역 3곳(3107~3109)의 집계도 비어 있습니다(3,888행).
- NEIS는 2025년 1~2월 일정을 반환하지 않았습니다. 이 기간은 1년 뒤 같은 날짜 값으로 대체했습니다(`school_vac_imputed`).
- NEIS 적재일이 2026-11로 찍혀 있어 학기 중 일정 변경 여부는 확인할 수 없습니다. 그래서 방학·재량휴업 같은 연간 계획 항목만 사용했습니다.
- 공휴일 출처는 대부분 뉴스와 블로그입니다. 독립 라이브러리와는 일치했지만, 월력요항 원문과 대조하면 더 확실합니다.

---

## 2. 파일 목록 — 무엇이 들어 있나

### 대회 제공 원본 (수정하지 않음)
| 파일 | 행 × 열 | 내용 |
|---|---|---|
| `train.csv` | 688,176 × 5 | `date, station_id, hour, boardings, alightings`. 2025-01-01~2026-06-30, 6~23시(23 = 23~24시). 신규역 3곳은 2025-06-28부터 |
| `test.csv` | 157,194 × 4 | `example_id, date, station_id, hour`. 2026-07-01~10-31, 정답 없음 |
| `sample_submission.csv` | **117,576** × 2 | `example_id, boardings(=0)`. **잘린 파일, 0번 참고** |
| `metaData.csv` | 71 × 5 | `station_id, station_name, canonical_line(1·2·7), first_date, last_date`. 환승역은 노선마다 다른 ID (예: 인천시청 3124·3221) |

### 실행 코드 (.py)
| 파일 | 하는 일 | 입력 → 출력 |
|---|---|---|
| `build_calendar.py` | 2025-01-01~2026-10-31 일별 달력 생성. 공휴일 40일, 연휴 블록, 징검다리, 방학, 여름휴가 극성수기 | (`external/school_calendar.csv` 있으면 사용) → `calendar_features.csv` |
| `fetch_neis.py` | NEIS Open API로 인천 초·중·고 557곳 학사일정 수집. **인증키 필요, 한 번만 실행** | `neis_key.txt` → `external/neis_raw/*.json` |
| `build_school_calendar.py` | 학교별 방학일을 판정해 날짜별 "방학 중 학교 비율" 계산 | `external/neis_raw/` → `external/school_calendar.csv` |
| `build_baseball.py` | KBO 일정 문서 5종 + 과거 결과로 문학 SSG 홈경기 CSV 생성, `games_known_at(cutoff, start, end)`로 기준일에 공개된 정보만 반환 | `external/kbo_raw/` → `external/munhak_games.csv`, `munhak_schedule_coverage.csv` |
| `features.py` | 기준일까지의 실측만으로 대상 기간 피처 생성. `build_features(cutoff, start, end)`, 합의된 기준일 3개는 `SPLITS` | train·test·meta + 달력 → `features_{valid_main,valid_aux,final}.csv` |

실행 순서 (처음부터 재현할 때):
```bash
python build_school_calendar.py   # neis_raw가 이미 있으므로 fetch_neis.py는 생략 가능
python build_calendar.py
python features.py --split all
```

### 생성 데이터 (.csv)
| 파일 | 행 × 열 | 내용 |
|---|---|---|
| `calendar_features.csv` | 669 × 44 | 하루 1행. 날짜 파생(연·월·요일·주차), 공휴일 이름·종류, 대체·임시·명절·선거, 휴무 블록 길이·순번, 연휴 전후·징검다리, 휴무일까지 거리, 초중고·대학 방학, 여름 극성수기, NEIS 학교 비율 5종. 정의는 `CALENDAR_SOURCES.md` 3절 |
| `external/school_calendar.csv` | 669 × 7 | 하루 1행. `school_vac_share`(전체), `_elem/_mid/_high`(학교급별) 방학 학교 비율, `school_discretionary_share`(평일 재량휴업 비율), `school_vac_imputed` |
| `external/munhak_games.csv` | 311 × 11 | 한 행 = (경기, 정보 출처). `game_date, start_time_kst, matchup, game_type(exhibition/regular/postseason), status(scheduled/played/cancelled/reserve/time_change), time_source(rule/actual/announced), source, published_date, status_known_date, source_url, note` |
| `external/events.csv` | 26 × 10 | 야구 외 대형 행사 목록: 시작·종료일, 행사장, 행사명, 분류(concert/festival), 시작 시각, 역 ID, 출처 URL, 비고 (`build_events.py`) |
| `external/events_daily.csv` | 82 × 8 | 위 행사를 날짜 × 역 단위로 펼친 것 (역 매핑 없는 행사는 station_id=0) |
| `external/weather_hourly.csv` | 64,224 × 27 | 4개 권역 × 2025-01-01~2026-10-31 시간별 날씨 + 일 단위 요약·플래그 (`build_weather.py`, Open-Meteo) |
| `external/weather_zone_map.csv` | 71 × 2 | 역 → 날씨 권역 |
| `external/kbo_crowd/crowd_munhak.csv` | 144 × 6 | 문학 정규시즌 경기별 관중 수 (2025·2026) |
| `external/munhak_schedule_coverage.csv` | 5 × 4 | 일정 문서별 공개일과 다루는 날짜 범위 (범위 밖 미래 날짜는 미공개 → NaN) |
| `external/kbo_raw/` | docx 5 + json 13 | KBO 보도자료 별첨 일정표 원본, SSG 경기 결과 (2026은 3~6월만) |
| `external/neis_raw/` | 557개 JSON, 17MB | NEIS 원본 (학교 정보 + 일정: 날짜, 행사명, 수업공제일 구분). 키 미포함 |
| `features_valid_main.csv` | 157,194 × 72 | 2025-06-30 기준 → 2025-07~10. 정답 포함 |
| `features_valid_aux.csv` | 157,194 × 72 | 2026-02-27 기준 → 2026-02-28~06-30. 정답 포함 |
| `features_final.csv` | 157,194 × 71 | 2026-06-30 기준 → test 행 + `example_id` |

피처 72개 구성:
- 키 3개
- 정답 2개
- 달력 32개
- 학교 5개
- 역 정보 4개
- horizon 1개
- 역×요일×시간 승차 집계 9개(전체·평상일·최근 8주 × 평균·log 평균·건수)
- 하차 요약 7개
- 전년 대비·추세 3개
- 작년 같은 기간 3개
- 364일 전 실측 3개

컬럼별 상세는 `FEATURES.md`에 있습니다.

### 문서 (.md)
| 파일 | 내용 |
|---|---|
| `CALENDAR_SOURCES.md` | 외부 자료 출처·조회일, 방학 판정 규칙, 데이터 대조 결과, 달력 컬럼 정의 |
| `FEATURES.md` | `features.py` 사용법, 피처 정의, 단일 피처 RMSLE, 주의사항 |
| `DATA_AUDIT.md` | 이 문서 |

### 제출물에서 빼야 할 것
- `neis_key.txt`: 개인 인증키. `.gitignore`에 등록해 두었습니다.
- `features_*.csv`: 66~81MB. 코드로 다시 만들 수 있으므로 제외하는 것을 권장합니다.
