# SINCON 2026 핫식스 월요일 보정 2단계

1단계 `submission_clean_refined.csv`에 `timecapsule → plave_existing → station025 → friday → monday → plave22 → lunar3122`를 적용해 `submission_monday_main.csv`를 만듭니다. 모델 재학습·제출 기능은 없습니다. 월요일은 2026-08-17·2026-10-05 두 날짜에만 적용하며 금요일의 제외 역·보호 행·상수·산식을 유지합니다. 기존 6개 층 코드와 상수는 원문 그대로입니다.

## Kaggle

1. 기존 1단계 `final_package/SINCON_final.ipynb`를 `RUN_MODE="assemble"`로 실행합니다.
2. 이 ZIP을 입력 Dataset에 풀고 공식 CSV 세 개를 별도 연결합니다. 1단계 CSV 경로를 `BASELINE_CSV`로 지정합니다.
3. `SINCON_monday.ipynb`의 `ROOT`를 패키지 폴더로 지정해 모든 셀을 실행합니다.
4. `/kaggle/working/sincon_monday_시각/`의 CSV와 `verification.json`을 확인합니다. 업로드는 하지 않습니다.

## Colab

1. 기존 1단계 노트북을 assemble로 실행합니다.
2. ZIP을 `/content` 아래에 풀고 공식 CSV를 `data/`에, 1단계 CSV를 `input/`에 놓습니다.
3. `ROOT`를 지정해 월요일 노트북 모든 셀을 실행합니다.
4. `/content/sincon_monday_시각/`의 CSV와 검산 결과를 확인합니다.

## 로컬

1. Python 3.11 또는 3.12와 requirements.txt의 NumPy·pandas를 준비합니다. 측정 버전은 아래 표에 있습니다.
2. 공식 CSV 세 개를 `data/`에, 1단계 출력 CSV를 `input/`에 놓습니다.
3. 노트북에서 `ROOT`·`BASELINE_CSV`를 지정해 전 셀을 실행하거나 `python -B execute_notebook.py . --output ../monday_run_01`을 실행합니다.
4. `verification.json`에서 후보 및 monday 비활성화 대조를 확인합니다. 기존 실행 폴더는 덮어쓰지 않습니다.

Kaggle·Colab 안내는 실행 절차입니다. 실기 검증은 macOS arm64 로컬에서만 했습니다. 1단계 재학습·클라우드·Linux 실기를 새로 수행하지 않았습니다.

## 입력과 출력 계약

공식 CSV는 official_inputs.json의 해시를 검사합니다. ROOT/data를 먼저 보고, `/content` 깊이 3·drive 제외, `/kaggle/input` 깊이 4까지만 탐색합니다. 심볼릭 링크를 따라가지 않습니다. 패키지는 고유 `monday_delivery_manifest.json`과 `package_kind=sincon_monday_holiday_preregistered_v1`으로 구분합니다. 후보가 여러 개이면 ROOT를 요구합니다.

1단계 출력은 157,194행·고유 example_id·공식 순서·유한·비음수여야 합니다. 동봉 기준과 최대 log1p 차이가 1e-9 이하면 수치 동등 입력으로 허용합니다. 이때 입력 차이와 SHA를 기록하고 **동봉 기준 표현으로 정규화한 뒤** 계산합니다. 기존 6개 층의 정확한 해시 검사와 원문 코드를 보존하기 위한 정책입니다. 9e-10 변화를 준 입력도 아래 로컬 검사에서 통과했습니다. input/이 비어 있으면 reference/ 기준을 사용합니다. 출력 후보 허용 오차는 최대 log1p 1e-10입니다.

패키지와 /kaggle/input은 읽기 전용 입력입니다. 출력은 Kaggle /kaggle/working, Colab /content, 로컬 패키지 밖의 새 폴더입니다. OUTPUT_DIR로 별도 지정할 수 있습니다. seed가 쓰이는 확률 계산은 없으며 CPU 라이브러리별 1스레드입니다.

## 검증표

| environment | case | python | numpy | pandas | code_cells | max_abs_log1p_difference | monday_disabled_max_abs_log1p_difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| py311 | reference | 3.11.17 | 2.2.6 | 2.2.3 | 6 | 0 | 0 |
| py311 | numeric_equivalence | 3.11.17 | 2.2.6 | 2.2.3 | 6 | 0 | 0 |
| py312 | reference | 3.12.15 | 2.5.4 | 3.1.1 | 6 | 0 | 0 |
| py312 | numeric_equivalence | 3.12.15 | 2.5.4 | 3.1.1 | 6 | 0 | 0 |

위 4개 실행 모두 노트북 코드 6개 셀을 순차 compile/exec로 실행했습니다. Jupyter 커널 실행은 아닙니다. 읽기 전용 복사본에서 실행했고 원본 프로젝트 읽기·외부 쓰기·소켓을 차단했습니다. 입력 파일 해시는 실행 전후 같습니다. 최종 ZIP을 빈 폴더에 푼 별도 실행의 증거와 ZIP SHA는 배포본 옆 DISTRIBUTION.json 및 night14 보고서에 기록합니다.

기존·중첩 각 9개 기준일 중 발동 4개 전부 최종 전체 RMSLE가 개선했습니다. 나머지 5개는 변화 0입니다. **서로 다른 검증 날짜는 2026-03-02·2026-05-25 두 개뿐**이며 기준일의 평가기간은 겹칩니다. 새로운 독립 홀드아웃이 아닙니다. 상세표는 evidence/validation_scores.csv입니다. 공식 실측은 2026-06-30까지만 썼습니다.

최종 후보는 주후보와 두 대상 날짜 각각 1,008행, 총 2,016행만 다릅니다. 최대 절대 log 변화는 0.05, 시간대별 역 log 변화 합의 최대 절댓값은 4.88498130835e-15입니다. count 합이 0이라는 뜻은 아닙니다. 제외 역·보호 행 변화는 0입니다.

## 배포 범위와 남은 확인

ZIP은 코드·노트북·캐시·기준 CSV·검증표·출처표·FILE_LIST.tsv를 포함합니다. 공식 원본 CSV, 가상환경, wheel, 임시 실행 결과는 제외합니다. DISTRIBUTION.json은 ZIP 해시를 담는 별도 파일이며 ZIP 내부에는 넣지 않습니다. FILE_LIST.tsv와 DISTRIBUTION.json 자체·ZIP 자체는 자기 해시 목록에서 제외합니다.

외부 자료의 출처·이용 조건은 기존 SOURCES.md·LICENSE_NOTICES.md·DOCUMENT_CORRECTIONS.md를 보존했습니다. 그 문서의 과거 조회·수치·권리 판단을 이번 작업에서 새로 검증한 것은 아닙니다. PLAVE 예매 원출처·일부 외부 자료 권리는 기존 보류 상태입니다. 새 웹 조회·다운로드·Public 점수 이용·Kaggle 업로드는 하지 않았습니다.
