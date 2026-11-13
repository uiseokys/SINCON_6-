# SINCON 2026 외부 자료 출처 및 이용 조건

작성·온라인 재확인 기준일은 2026-11-12 KST다. `최초 조회 기록 없음`은 팀 문서와 보존 파일에서 실제 조회일을 확인할 수 없다는 뜻이다. ZIP 생성일, CSV 수정일, 기사 게시일은 최초 조회일로 옮기지 않았다. 이용 조건의 확인 여부는 대회 운영진의 최종 판단과 별개다.

## 적용 범위

원본 모델은 팀의 LightGBM·임베딩 MLP에서 시작한다. `sincon_local_candidate.py`와 `sincon_conditional_candidate.py`가 구 단위 비지하철 이동량, 버스, 경기, 축제, 주거 변화를 추가한다. `sincon_clean_15126_replay.py`가 `submission_local_stack.csv`에서 검증 기반 보정을 재계산한다. 후속 `sincon_clean_event_replay.py`, `sincon_clean_model_layers.py` 및 TabICL·벡터 XGBoost는 아래 외부 피처와 공지를 참조한다. `train.csv`의 승차 관측이 학습과 시간순 검증의 정답이다. `test.csv`는 `example_id`, 날짜, 역, 시각을 제공한다. [대회 데이터와 설명](https://www.kaggle.com/competitions/sincon-2026/data)

## 원본 모델과 전이 모델에 들어간 자료

| 자료와 사용처 | 조회 기록 | 이용 조건과 원문 |
| --- | --- | --- |
| 법정공휴일·대체공휴일·선거일. `calendar_features.csv` 및 팀 `make_calendar.py`의 휴일과 연휴 구분 | 최초 2026-11-11 (`external_inputs/CALENDAR_SOURCES.md`) · 원문 재확인 2026-11-12 | 법령·공식 달력에 공개된 날짜를 코드로 변환했다. 달력의 표현물과 이미지는 사용하지 않는다. [공휴일에 관한 법률](https://www.law.go.kr/법령/공휴일에관한법률) · [관공서 공휴일 규정](https://law.go.kr/LSW/lsLinkCommonInfo.do?chrClsCd=010202&lsJoLnkSeq=1018770105) · [한국천문연구원 월력요항](https://astro.kasi.re.kr/life/post/calendardata) |
| 나이스 인천 초·중·고 학사일정. `school_calendar.csv`의 날짜별 방학·재량휴업 비율 | 최초 2026-11-11 (`CALENDAR_SOURCES.md`) · 조건 재확인 2026-11-12 | 공공데이터포털은 이용허락범위 제한 없음으로 표기한다. 공식 API의 인증키·호출 제한을 적용한다. [학사일정 데이터와 이용 조건](https://www.data.go.kr/data/15137088/openapi.do) · [NEIS API](https://open.neis.go.kr/hub/SchoolSchedule) · [NEIS 이용약관](https://open.neis.go.kr/portal/userAgreementPage.do) |
| 대학 방학. `is_univ_vacation` 및 대학 방학 관련 플래그 | 최초 2026-11-11 (`CALENDAR_SOURCES.md`) · 법령 링크 재확인 2026-11-12 | 법령과 대학 공지의 학사일정 날짜를 사용한다. 단일 대학 자료와 규칙에 따른 대리 변수다. 인천대 첨부문서의 별도 라이선스는 확인 기록이 없다. [고등교육법 시행령](https://www.law.go.kr/법령/고등교육법시행령) · [인천대 2026 학사 공문](https://liberaledu.inu.ac.kr/bbs/inu/246/379145/download) |
| Open-Meteo 인천 시간별 날씨. 팀 원본은 인천 중심 1지점, `sincon_local_repro.zip`은 4권역 격자와 역별 매핑 | 최초 2026-11-11 (`CALENDAR_SOURCES.md`) · 조건 재확인 2026-11-12 | 공개 날씨 데이터는 CC BY 4.0. Open-Meteo·Copernicus/ECMWF를 표시하고 권역 매핑·일 집계 등 가공 내용을 밝힌다. 무료 API의 비상업적 이용과 호출 한도를 지킨다. [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) · [API 조건](https://open-meteo.com/en/features#terms) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| 서울시·KT 수도권 생활이동 **OA-22655**. 2025-01~2026-10 인천·부천 구 단위로 가공한 `mobility_monthly/`, LightGBM·일별 신호·TabICL 및 후속 벡터 입력 | 최초 조회 기록 없음 · 데이터 페이지·조건 재확인 2026-11-12 | 서울시 공식 메타데이터는 공공누리 제1유형, 저작권자 서울특별시, 제3저작권자 없음으로 명시한다. 출처 표시 조건으로 이용·변경 가능하다. 수집 코드가 광역버스(4), 일반버스(5), 도보(7), 차량(8)만 먼저 골라 집계한다. 동결 ZIP의 22개 파일, 779,022행을 읽어 네 범주와 2025-01-01~2026-10-31 범위를 확인했다. [OA-22655 원본과 조건](https://data.seoul.go.kr/dataList/OA-22655/F/1/datasetView.do) · [공공누리 제도](https://www.mcst.go.kr/site/s_open/kogl/intro.jsp) |
| 인천교통공사 역 주소, 출입구 좌표. 구 매핑, 역 중심 계산과 버스·주거 연결 | 최초 조회 기록 없음 · 조건 재확인 2026-11-12 | 두 항목 모두 포털의 이용허락범위 제한 없음. 역 주소와 위치 정보만 활용한다. [역 주소 15043811](https://www.data.go.kr/data/15043811/fileData.do) · [출입구 좌표 15166820](https://www.data.go.kr/data/15166820/fileData.do) · [포털 정책](https://www.data.go.kr/ugs/selectPortalPolicyView.do) |
| 인천 버스 정류장별 월간 승하차 및 전국 버스 정류장 좌표. 인접 버스 이용량, 검단권 변화량 | 최초 조회 기록 없음 · 조건 재확인 2026-11-12 | 두 항목 모두 포털의 이용허락범위 제한 없음. 버스 정류장 자료다. [인천 버스 15048264](https://www.data.go.kr/data/15048264/fileData.do) · [정류장 좌표 15067528](https://www.data.go.kr/data/15067528/fileData.do) · [포털 정책](https://www.data.go.kr/ugs/selectPortalPolicyView.do) |
| 서울 버스 정류장별 승하차 **OA-12912**. `sincon_seoul_bus_daily.csv.gz`, 별도 연결 버스 실험의 입력 | 최초 조회 기록 없음 · 조건 재확인 2026-11-12 | 공공누리 제1유형. 서울시와 자료명을 표시한다. 검증 기반 `local_stack`과 후속 재실행 경로는 이 별도 파일을 직접 읽지 않는다. [OA-12912 원본과 조건](https://data.seoul.go.kr/dataList/OA-12912/F/1/datasetView.do) |
| K-apt 단지 사용승인일·최초입주월·세대수·좌표. `sincon_housing_approvals.csv`의 35개 단지 | 최초 조회 기록 없음 · 공개 포털 조건 재확인 2026-11-12 | 포털의 국토부·한국부동산원 기본정보는 이용허락범위 제한 없음으로 공개한다. 동결 파일의 실제 수집 코드는 K-apt 웹 지도·단지 조회를 호출했다. 지도 응답의 최초입주월·좌표까지 포털 배포본과 동일한 이용허락이 적용되는지 확인 기록이 없다. [K-apt 실제 출처](https://www.k-apt.go.kr/kaptinfo/openKaptLocation.do) · [국토부 단지 기본정보 15073271](https://www.data.go.kr/data/15073271/fileData.do) · [한국부동산원 단지 식별정보 15106861](https://www.data.go.kr/data/15106861/fileData.do) |
| OpenStreetMap 기반 보행거리. `sincon_housing_walking_distance.csv`, 별도 도보거리 주거 실험의 입력 | 최초 조회 기록 없음 · OSM 라이선스 재확인 2026-11-12 | ODbL. OpenStreetMap 기여자 표기가 필요하고, 파생 데이터베이스 배포 시 동일 조건 적용 범위를 확인한다. 사용한 Valhalla 공개 서비스의 별도 조건은 확인하지 못했다. 검증 기반 `local_stack`의 거리 계산은 역과 단지 좌표를 사용한다. [OSM 저작권 및 이용허락](https://www.openstreetmap.org/copyright) · [실제 Valhalla endpoint](https://valhalla1.openstreetmap.de/sources_to_targets) |
| 문체부 지역축제 개최계획 및 전국문화축제표준데이터. `sincon_festival_station_days.csv`의 날짜·장소와 전년 방문자 대리값 | 최초 조회 기록 없음 · 배포 페이지 재확인 2026-11-12 | 표준데이터에는 포털 정책 링크가 표시된다. 사용한 문체부 연도별 첨부파일 자체의 개별 공공누리 유형은 확인 기록이 없다. 출처 표기에 문체부와 한국관광공사·제공 지자체를 포함한다. [문체부 지역축제](https://www.mcst.go.kr/site/s_culture/festival/festivalList.jsp?pMenuCD=0304000000) · [축제표준데이터 15013104](https://www.data.go.kr/data/15013104/standard.do) |
| KBO 문학 경기 일정·실제 결과·관중과 경기 종료. 팀 `munhak_games.csv`, `crowd_munhak.csv`, 별도 `sincon_kbo_game_data.csv` 160행 | 팀 최초 2026-11-11 (`CALENDAR_SOURCES.md`) · 별도 게임센터 최초 조회 기록 없음 · 조건 재확인 2026-11-12 | **수집 및 이용 허용 미확인.** 사이트 약관 제14조 차항은 사전 서면 동의 또는 공식 API 없이 회원이 자동화 도구로 대량·반복 수집하는 행위를 금지한다. 같은 조항에서 재배포·타 플랫폼 연동·2차 가공도 제한한다. 별도 160행은 `sincon_kbo_game_data.py`가 사이트의 `/ws/Schedule.asmx/` 경로에서 병렬 수집했다. 이 경로가 약관상 공식 API인지, 권리자의 허락을 받았는지 확인 기록이 없다. 팀이 관중 표를 수집한 방식도 확인 기록이 없다. [KBO 경기일정](https://www.koreabaseball.com/Schedule/Schedule.aspx) · [시즌 일정 보도자료](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=11794) · [관중 현황](https://www.koreabaseball.com/Record/Crowd/GraphDaily.aspx) · [KBO 이용약관](https://www.koreabaseball.com/Etc/Policy.aspx) |
| 한국프로축구연맹 부천종합운동장 경기·킥오프·관중, `sincon_bucheon_football_data.csv` 39행 | 최초 조회 기록 없음 · 사이트와 약관 재확인 2026-11-12 | **이용 허용 미확인.** `sincon_bucheon_football_data.py`가 공식 홈페이지의 `getScheduleList.do`로 월별 요청을 보냈다. 공식 재사용 API 또는 이 자료의 비상업적 학습·재배포를 허용하는 라이선스는 확인 기록이 없다. [K리그 부천 구단](https://www.kleague.com/club/club.do?teamId=K26) · [K리그 홈페이지 약관](https://www.kleague.com/join/step1.do) |
| 예술경영지원센터 KOPIS 공연·장소·날짜·정원. `sincon_kopis_scheduled.csv`를 사용한 별도 공연 모델은 미채택했다. 원본 팀 수기 행사표의 두 공연은 아래 KOPIS 공개 페이지를 출처로 기록했다. | 최초 조회 기록 없음 · 운영기관·API 안내 재확인 2026-11-12 | 별도 실험의 **웹 수집 경로 이용 조건 미확인**. 공식 Open API는 인증키가 필요하다. 실제 수집 스크립트는 웹 서비스의 `:9001/api/prs/v1/por/db`를 사용했다. 공식 Open API의 이용조건을 해당 웹 경로에 그대로 적용할 근거는 확인되지 않았다. [KOPIS 공연 목록](https://kopis.or.kr/por/db/pblprfr/pblprfr.do?menuId=MNU_00020) · [운영기관 및 근거](https://kopis.or.kr/mob/cs/csInfo.do) · [공식 API 안내](https://www.kopis.or.kr/por/cs/openapi/openApiFaq.do?menuId=MNU_00074) |
| 팀 수기 행사 일정, `external/events.csv`에서 `events_daily.csv` 생성. 팀 원본 및 동결 ZIP의 행사 피처 | 팀 매니페스트 공통 최초 조회일 2026-11-11 · 개별 행사 조회일은 확인 기록 없음 | 예매처·주최 측·지자체에서 공개한 날짜, 장소, 시작 시각 등의 사실을 입력한다. 페이지의 개별 라이선스는 아래 목록을 포함해 일괄 확인되지 않았다. 주최 측 공지와 티켓 페이지의 포스터, 이미지, 기사 본문은 모델 입력에 포함하지 않는다. [동결 출처 표](#팀-원본-행사-출처) |

`share_to_modeler.zip`의 최신 `arr_*`/`mob_gu`는 별도 피처다. 여기서 확인한 사용 경로는 `sincon_mobility_data.py`의 교통수단 필터를 통과한 OA-22655 집계다. 버스·도보·차량과 같은 구 단위 유동인구에 대한 서울시·KT 추정치이며, 특정 역의 승하차 실측과 합산하지 않는다. Meteostat는 팀에서 날씨 대조에만 썼고, 보존된 원본·후속 예측 피처의 날씨는 Open-Meteo다.

## 사전학습 모델

| 구성요소 | 조회 기록 | 이용 조건과 원문 |
| --- | --- | --- |
| `tabicl==2.2.0`의 `TabICLRegressor` 추론 패키지 | 최초 조회 기록 없음 · 설치 메타데이터와 공식 배포 페이지 확인 2026-11-12 | 핵심 코드는 BSD-3-Clause. 저작권 문구와 조건을 보존한다. 프로젝트의 선택적 `forecast` 서브모듈에는 별도 Apache-2.0 공지가 있다. 이번 회귀 경로는 핵심 `TabICLRegressor`를 사용한다. [정확한 PyPI 버전](https://pypi.org/project/tabicl/2.2.0/) · [프로젝트 LICENSE](https://github.com/soda-inria/tabicl/blob/main/LICENSE) |
| `jingang/TabICL`의 `tabicl-regressor-v2-20260212.ckpt` | 최초 조회 기록 없음 · 파일별 라이선스와 해시 확인 2026-11-12 | 정확한 체크포인트의 공식 파일 페이지에 BSD-3-Clause가 표시된다. 공개 SHA-256 `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`가 로컬 체크포인트와 일치한다. [체크포인트 파일과 해시](https://huggingface.co/jingang/TabICL/blob/main/tabicl-regressor-v2-20260212.ckpt) · [모델 카드](https://huggingface.co/jingang/TabICL) |

LightGBM·XGBoost와 임베딩 MLP·그래프 신경망은 이 경로에서 대회 `train.csv`로 학습한 모델이다. 외부 학습 가중치를 사용하는 구성요소는 위 TabICL이다.

## 후속 보정에 사용한 개별 공지

이 표에는 일정의 날짜, 시각, 장소, 취소, 교통 안내, 경기·공연 사실을 확인한 페이지가 들어 있다. 아래 출처의 조회일은 개별 기록이 남아 있을 때만 적었다. 공개 공지의 본문·포스터·영상·스크린샷 재배포 허락이 확인되었다는 뜻은 아니다. 별도 표시가 없는 페이지의 포괄적 라이선스는 확인 기록이 없다.

| 보정 | 확인한 내용과 출처 |
| --- | --- |
| 문학경기장 취소 72행 | OBJET의 취소 공지는 주최·출연팀 공지에 근거한다. GHOSTIVAL은 예매 페이지와 취소 보도에 근거한다. 최초 조회 기록 없음. [OBJET 주최 측](https://btobcompany.kr/notice/?bmode=view&category=&idx=174155451) · [OBJET 출연팀](https://weverse.io/qwer/notice/39354) · [GHOSTIVAL 예매처](https://nol.yanolja.com/ticket/products/26014180) · [GHOSTIVAL 취소 보도](https://www.asiatoday.co.kr/kn/view.php?key=20261021010007097) |
| 송도해변축제 장소·저녁 보정 | 송도달빛공원 개최 날짜·장소와 2025년 행사 중단일을 대조했다. 최초 조회 기록 없음. [2026 연수구 개최 결과](https://www.yeonsu.go.kr/main/community/notify/report.asp?page=v&seq=29334) · [2026 주최기관 프로그램](https://www.ysfac.or.kr/user/cultural/events_view.php?sq=381) · [2025 연수구 결과](https://www.yeonsu.go.kr/main/community/notify/report.asp?page=v&seq=28076) · [2025 연수구 중단 공지](https://www.yeonsu.go.kr/main/community/notify/notice.asp?page=v&seq=114222) |
| 기존 행사 일정·접근역 조정 | 송도맥주축제 운영 기간, 부천국제만화축제의 산업행사 분리, 부평풍물대축제 금요일 행사 위치. 최초 조회 기록 없음. [송도맥주축제 주최 측](https://www.songdobeer.com/) · [부천시 2025 만화축제](https://news.bucheon.go.kr/home/kor/M771882326/board.do?act=detail&deleteAt=N&idx=cb00d1a0008a85e71a41b8741facbffe035ef3c02298252718467b719454e8f2) · [부천시 2026 만화축제](https://news.bucheon.go.kr/home/kor/M771882326/board.do?act=detail&deleteAt=N&idx=cb00d1a0008a85e71a41b8741facbffee67a5aea52d441be34a8b95e07a215d7) · [부평구 2025 행사 위치](https://www.icbp.go.kr/bbs/bbsMsgDetail.do?bcd=media_photo&keyfield=title&msg_seq=43619&pgdiv=headman&pgno=71) · [부평구 2026 행사 위치](https://www.icbp.go.kr/dong/bbs/bbsMsgDetail.do?bcd=dong_photo&dong=cheongcheon2&msg_seq=17006) |
| 문학경기장 관람 경기 및 우천 취소 | SSG 특별경기·불꽃·방송과 취소 시각을 확인했다. 과거 대조 기사 URL은 `sincon_munhak_spectator.py`, `sincon_kbo_cancellation_time.py`의 `HISTORY`에 보관했다. 최초 조회 기록 없음. [SSG 구단 공지](https://www.ssglanders.com/media/news/detail?idx=19944) · [2026 관람 경기 공식 방송](https://www.youtube.com/live/TzunCSawrp4) · [8월 22일 취소 보도](https://www.mk.co.kr/news/sports/12133816) · [9월 25일 취소 보도](https://m.mk.co.kr/amp/12161370) |
| 아시아드 싸이 공연·과학대제전 | 7월 공연 장소·시각 및 10월 과학대제전 실제 개최·프로그램·교통. 최초 조회 기록 없음. [싸이 예매 안내](https://ticket.interpark.com/OtherSite/Ticket/Ticket_GoodsInfo.asp?Ca=01003&GroupCode=26007871) · [아시아드 공식 접근 안내](https://www.insiseol.or.kr/sport/asiad/introduce/location.jsp) · [2025 교육청](https://www.ice.go.kr/ice/na/ntt/selectNttInfo.do?mi=11620&nttSn=3343366) · [2025 인천대](https://www.inu.ac.kr/bbs/inu/2989/414648/artclView.do) · [2026 교육청 결과](https://www.ice.go.kr/ice/na/ntt/selectNttInfo.do?mi=&nttSn=3387304) · [2026 프로그램](https://www.icemf.kr/) |
| 원미구민 체육대회 | 2025년과 2026년 행사 날짜·장소와 종합운동장 접근. 최초 조회 기록 없음. [2025 부천시](https://news.bucheon.go.kr/home/kor/M749153916/board.do?act=detail&deleteAt=N&idx=cb00d1a0008a85e71a41b8741facbffe449f3b764132bc8324aebf94beb102a6) · [2026 원미구 일정](https://wonmi.bucheon.go.kr/site/program/board/basicboard/view?boardid=1209247&boardtypeid=28300&menuid=170005017006001&pagesize=10) · [2026 부천시 개최 결과](https://news.bucheon.go.kr/home/kor/M749153916/board.do?act=detail&deleteAt=N&idx=cb00d1a0008a85e71a41b8741facbffe431bafabae2b14835799748c3826c8dd) · [종합운동장 교통 안내](https://www.best.or.kr/fmcs/289) |
| 동춘 대형마트 휴무 | 일요일 중 둘째·넷째 주에 정기 휴점한다는 근거다. 예외 일자는 상점의 실제 방문 기록이 아닌 달력 규칙으로 정했다. 원문 재확인 2026-11-12. [연수구 무인발급기 운영표의 이마트 연수점 휴무](https://www.yeonsu.go.kr/main/general/civil/manless.asp) |
| 검암 드림파크 국화축제 | 2025·2026 행사, 검암역 버스 접근, 강풍·폭우 중단일 제외. 최초 조회 기록 없음. 중단 여부 보조 자료는 민간 사이트다. [2025 주최기관](https://dreamparkcf.com/sub/board.php?bn=hana_board_16&fn=setting&gb=hana_board_16&mn=business&zest_bn=hana_board_15) · [2026 한국관광공사 일정](https://korean.visitkorea.or.kr/kfes/detail/fstvlDetail.do?cmsCntntsId=3377402) · [교통 안내](https://slc.or.kr/slc/sli/di/direct01.do) · [중단 보조 자료](https://trip.beomtachi.com/dream-park-chrysanthemum-festival) |
| 선학 능허대축제 | 2026 공연 종료와 선학역 접근, 2025 행사 및 선학권 공연 비교. 최초 조회 기록 없음. [연수문화재단 프로그램](https://www.ysfac.or.kr/user/cultural/events_view.php?search=&sq=400) · [연수구 교통](https://www.yeonsu.go.kr/tour/festival/neungheodae_festival/location.asp) · [2025 연수구 개최 결과](https://www.yeonsu.go.kr/etc/bbs/SearchView.asp?bbs_code=board_3&seq=28296) · [2025 선학 개관공연](https://www.yeonsu.go.kr/culture/show/art_show.asp?idx=827&page=v) · [문학 비교 공연](https://weverse.io/bts/notice/30499?hl=ko) |
| 인천역사문화둘레길 | 2025·2026 행사 개최 및 일정·접근. 최초 조회 기록 없음. [2025 인천시 행사](https://www.incheon.go.kr/mayor_8/MA040101/view?repSeq=DOM_0000000013314801) · [2026 인천시 행사](https://www.incheon.go.kr/IC010205/view?repSeq=DOM_0000000015868742) · [부평구 일정](https://www.icbp.go.kr/dong/bbs/bbsMsgDetail.do?bcd=dong_notice&dong=cheongcheon2&msg_seq=136242&pgno=1) |
| 검단 연장 개통 | 검단연장선 개통일을 기준으로 개통 전후 수요를 대조한다. 최초 조회 기록 없음. [인천시 개통 안내](https://www.incheon.go.kr/mayor_8/MA040101/view?curPage=255&repSeq=DOM_0000000012620245) |

## 팀 원본 행사 출처

아래는 `sincon_local_repro.zip`에 동결한 `external_inputs/external/events.csv`의 출처다. 표의 공통 최초 조회 기록은 2026-11-11이며, 행사별 날짜는 따로 남아 있지 않다.

| 행사 | 일정 | 원본 기록 |
| --- | --- | --- |
| SEVENTEEN in CARAT LAND 2025 | 2025-03-20 ~ 2025-03-21 | [팀 출처](<https://namu.wiki/w/인천문학경기장%20주경기장>) |
| YE LIVE IN KOREA | 2025-07-26 | [팀 출처](<https://namu.wiki/w/인천문학경기장%20주경기장>) |
| MUSE LIVE IN KOREA | 2025-09-27 | [팀 출처](<https://namu.wiki/w/인천문학경기장%20주경기장>) |
| 진 #RUNSEOKJIN_EP.TOUR ENCORE | 2025-10-31 ~ 2025-11-01 | [팀 출처](<https://www.etoday.co.kr/news/view/2520535>) |
| 박효신 LIVE A & E 2026 | 2026-04-04 ~ 2026-04-05 | [팀 출처](<https://livewire.so/places/ebKtT7uZ1dw/events>) |
| 박효신 LIVE A & E 2026 | 2026-04-11 | [팀 출처](<https://livewire.so/places/ebKtT7uZ1dw/events>) |
| Sustainable Wave Festival | 2026-08-29 ~ 2026-08-30 | [팀 출처](<https://livewire.so/places/ebKtT7uZ1dw/events>) |
| PLAVE World Tour KEEP IT MANIC | 2026-09-12 ~ 2026-09-13 | [팀 출처](<https://livewire.so/places/ebKtT7uZ1dw/events>) |
| OBJET K-POP FESTA 2026 | 2026-09-26 ~ 2026-09-27 | [팀 출처](<https://livewire.so/places/ebKtT7uZ1dw/events>) |
| GHOSTIVAL [인천] | 2026-10-24 ~ 2026-10-25 | [팀 출처](<https://livewire.so/places/ebKtT7uZ1dw/events>) |
| 싸이 흠뻑쇼 SUMMERSWAG 2025 인천 | 2025-06-28 ~ 2025-06-29 | [팀 출처](<https://namu.wiki/w/싸이%20흠뻑쇼%202025>) |
| SEVENTEEN WORLD TOUR NEW_ IN INCHEON | 2025-09-13 ~ 2025-09-14 | [팀 출처](<https://ddonitong.com/394>) |
| Stray Kids World Tour dominATE celebrATE | 2025-10-18 ~ 2025-10-19 | [팀 출처](<https://www.kihoilbo.co.kr/news/articleView.html?idxno=3000426>) |
| SEVENTEEN WORLD TOUR NEW_ 앵콜 | 2026-04-04 ~ 2026-04-05 | [팀 출처](<https://kopis.or.kr/mob/db/prfplcView.do?menuId=MNU_00021&mt10Id=FC002083>) |
| SEVENTEEN 10TH FAN MEETING CARAT LAND | 2026-06-20 ~ 2026-06-21 | [팀 출처](<https://kr.trip.com/blog/seventeen-kr-concert/>) |
| 싸이 흠뻑쇼 SUMMERSWAG 2026 인천 | 2026-07-11 | [팀 출처](<https://www.gukjenews.com/news/articleView.html?idxno=3632577>) |
| 인천펜타포트 락 페스티벌 2025 | 2025-08-01 ~ 2025-08-03 | [팀 출처](<https://magazine.hankyung.com/job-joy/article/202508064376d>) |
| 송도해변축제 2025 | 2025-08-09 ~ 2025-08-15 | [팀 출처](<https://news.tf.co.kr/read/ogmeta/2231440.htm>) |
| 송도맥주축제 2025 | 2025-08-22 ~ 2025-08-30 | [팀 출처](<https://www.heraldk.com/article/2025073015143693639>) |
| 제13회 연수 능허대 문화축제 | 2025-10-17 ~ 2025-10-19 | [팀 출처](<https://www.yeonsu.go.kr/tour/festival/neungheodae_festival/info.asp>) |
| 인천펜타포트 락 페스티벌 2026 | 2026-07-31 ~ 2026-08-02 | [팀 출처](<https://m.go.seoul.co.kr/news/2026/07/30/20260730500213>) |
| 송도해변축제 2026 | 2026-08-08 ~ 2026-08-15 | [팀 출처](<https://marketingnote.kr/2026-songdo-beach-festival-guide/>) |
| 송도맥주축제 2026 | 2026-08-22 ~ 2026-08-30 | [팀 출처](<https://songdobeer.com/>) |
| 이슬라이브 페스티벌 2026 | 2026-09-12 | [팀 출처](<https://kopis.or.kr/por/db/prfplc/prfplcView.do?menuId=MNU_00021&mt10Id=FC002007>) |
| 제14회 연수 능허대 문화축제 | 2026-10-09 ~ 2026-10-10 | [팀 출처](<https://www.reportera.co.kr/travel/yeonsu-neungheodae-cultural-festival/>) |
| 부천페스타 봄꽃 여행 / 원미산 진달래축제(4/4~5) | 2026-03-28 ~ 2026-04-05 | [팀 출처](<https://www.bucheon.go.kr/site/homepage/menu/viewMenu?menuid=145002006>) |
