# 외부 자료와 이용 조건

이 표는 2026-11-12에 팀이 작성·재확인한 출처 기록을 옮긴 것이다. 이번 패키지 작업에서는 새 외부 데이터를 수집하지 않았다. 최초 조회일이 없는 항목은 추측하지 않았다. 전체 개별 행사 URL과 원문 조건은 `docs/SINCON_EXTERNAL_SOURCES.md`에도 보존했다. 아래의 '7월 이후'는 2026년 7월 이후 정보다. 외부 날씨·버스·일정 등이 포함되지만, 지하철 승하차 실측은 포함하지 않는다.

| 이름·보관 위치·역할 | 출처 링크 | 조회일 | 이용 조건 | 7월 이후 정보 |
|---|---|---|---|---|
| 공휴일·선거일·연휴, `calendar_features.csv` | [법률](https://www.law.go.kr/법령/공휴일에관한법률), [월력요항](https://astro.kasi.re.kr/life/post/calendardata) | 최초 2026-11-11, 재확인 11-12 | 공개 날짜를 코드로 변환. 이미지·달력 표현물은 미사용 | 포함, 예정 날짜 |
| NEIS 학사일정, `school_calendar.csv` | [데이터포털](https://www.data.go.kr/data/15137088/openapi.do), [API](https://open.neis.go.kr/hub/SchoolSchedule) | 최초 2026-11-11, 재확인 11-12 | 포털 이용허락범위 제한 없음, API 호출 조건 적용 | 포함 |
| 대학 방학 대리변수 | [인천대 공문](https://liberaledu.inu.ac.kr/bbs/inu/246/379145/download), [법령](https://www.law.go.kr/법령/고등교육법시행령) | 최초 2026-11-11, 재확인 11-12 | 날짜 사실 사용. 대학 첨부문서의 별도 조건 미확인 | 포함 |
| Open-Meteo 시간별 날씨, 팀 1지점·후속 4권역 | [자료](https://open-meteo.com/en/docs/historical-weather-api), [이용 조건](https://open-meteo.com/en/features#terms) | 최초 2026-11-11, 재확인 11-12 | CC BY 4.0. Open-Meteo·Copernicus/ECMWF 출처 표기. 권역 매핑·집계 가공 | 포함, 실제 날씨 |
| 서울시·KT OA-22655 비지하철 이동량, `mobility_monthly/` | [자료·조건](https://data.seoul.go.kr/dataList/OA-22655/F/1/datasetView.do) | 최초 기록 없음, 재확인 2026-11-12 | 공공누리 제1유형. 서울특별시·KT 출처 표기. 광역버스4·일반버스5·도보7·차량8만 집계 | 포함, 실제 비지하철 이동량 |
| 인천교통공사 역 주소·출입구 좌표 | [역 주소](https://www.data.go.kr/data/15043811/fileData.do), [출입구](https://www.data.go.kr/data/15166820/fileData.do) | 최초 기록 없음, 재확인 2026-11-12 | 포털 이용허락범위 제한 없음. 주소·좌표만 사용 | 조회 시점의 정적 정보, 최초 갱신일 미확인 |
| 인천 버스 정류장별 월간 승하차·정류장 좌표 | [월간 버스](https://www.data.go.kr/data/15048264/fileData.do), [좌표](https://www.data.go.kr/data/15067528/fileData.do) | 최초 기록 없음, 재확인 2026-11-12 | 포털 이용허락범위 제한 없음 | 포함, 실제 버스 이용량 |
| K-apt 35개 단지 승인·입주월·세대수·좌표 | [실제 웹 출처](https://www.k-apt.go.kr/kaptinfo/openKaptLocation.do), [기본정보](https://www.data.go.kr/data/15073271/fileData.do) | 최초 기록 없음, 포털 조건 재확인 2026-11-12 | 포털 기본정보는 제한 없음. 실제 웹 지도 응답의 입주월·좌표에 같은 조건이 적용되는지 미확인 | 입주월에 이후 기간 포함. 지하철 관측 없음 |
| 문체부 지역축제 계획·축제 표준데이터·전년 관람객 대리값 | [문체부](https://www.mcst.go.kr/site/s_culture/festival/festivalList.jsp?pMenuCD=0304000000), [표준데이터](https://www.data.go.kr/data/15013104/standard.do) | 최초 기록 없음, 배포 페이지 재확인 2026-11-12 | 첨부파일별 공공누리 유형 미확인. 문체부·한국관광공사·제공 지자체 표기 | 포함, 일정·행사 사실 |
| KBO 문학 경기·실제 결과·관중·종료 시각 | [일정](https://www.koreabaseball.com/Schedule/Schedule.aspx), [관중](https://www.koreabaseball.com/Record/Crowd/GraphDaily.aspx), [약관](https://www.koreabaseball.com/Etc/Policy.aspx) | 팀 최초 2026-11-11, 별도 게임센터 최초 기록 없음, 조건 재확인 11-12 | 수집·이용 허용 미확인. 자동 수집·재배포 제한과 실제 경로의 관계 확인 필요 | 포함, 실제 경기 정보 |
| K리그 부천 경기·킥오프·관중 | [구단](https://www.kleague.com/club/club.do?teamId=K26), [약관](https://www.kleague.com/join/step1.do) | 최초 기록 없음, 재확인 2026-11-12 | 비상업 학습·재배포 허용 미확인 | 포함, 실제 경기 정보 |
| 팀 수기 공연·행사, `events.csv`, `events_daily.csv` | [주최 측 예시](https://www.songdobeer.com/), 행사별 전체 URL은 아래 부록 | 공통 최초 2026-11-11, 개별 조회일 기록 없음 | 일정·장소·시각 사실만 사용. 개별 페이지 이용 조건 미확인 | 포함, 일정·취소·개최 사실 |
| 취소·장소·접근·종료 등 후속 행사 공지 | [OBJET 주최](https://btobcompany.kr/notice/?bmode=view&category=&idx=174155451), [연수문화재단](https://www.ysfac.or.kr/user/cultural/events_view.php?sq=381), [과학대제전](https://www.icemf.kr/) | 최초 기록 없음, 팀 출처 정리 2026-11-12 | 공지의 사실만 입력. 개별 원문의 포괄적 이용 조건 미확인 | 포함, 실제 취소·위치·프로그램 등 |
| 동춘 대형마트 정기 휴무 | [연수구 운영표](https://www.yeonsu.go.kr/main/general/civil/manless.asp) | 재확인 2026-11-12 | 공식 표의 둘째·넷째 일요일 규칙을 사용. 페이지 개별 라이선스 미확인 | 정기 규칙의 예측 기간 적용 |
| TabICL 2.2.0 추론 코드 | [배포](https://pypi.org/project/tabicl/2.2.0/), [LICENSE](https://github.com/soda-inria/tabicl/blob/main/LICENSE) | 최초 기록 없음, 메타데이터 확인 2026-11-12 | 핵심 회귀 경로 BSD-3-Clause. 패키지에 포함한 라이선스 고지 보존 | 외부 사전학습 모델, 지하철 후속 실측 입력 없음 |
| TabICL 회귀 체크포인트 `20260212` | [공식 파일·해시](https://huggingface.co/jingang/TabICL/blob/main/tabicl-regressor-v2-20260212.ckpt) | 최초 기록 없음, 라이선스·해시 확인 2026-11-12 | 파일 페이지 BSD-3-Clause. SHA-256 `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a` | 체크포인트 이름은 2026-02-12. 학습 원자료 전체에 대한 별도 감사는 미실시 |

공식 `train.csv`는 2026-06-30까지 688,176행이고, `test.csv`는 157,194개의 날짜·역·시간·ID만 제공한다. 공식 CSV는 대회 참가자가 운영진 계정의 비공개 Dataset에 올리는 용도다. 일반 공개 재배포 허락으로 간주하지 않는다.

이용 조건 미확인 자료를 전부 제외한 결과 차이는 미계산이다. 보관 예측 안에 이미 그 자료의 영향이 들어 있으므로, 입력 CSV만 삭제한 조립을 자료 제거 실험으로 표시할 수 없다. KBO·K리그·주거·행사별 원본 모델과 검증 캐시까지 다시 학습해야 한다. 이번 패키지는 기준 정리본을 보존하며, 해당 자료의 규칙 3절 적합성을 보증하지 않는다. 후속 행사층만 뺀 조건부 차이는 별도 검증 보고서로 계산하며, 외부 자료 전체 제거 효과와 구분한다.
