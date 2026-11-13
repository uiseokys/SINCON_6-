"""외부 자료 특징: 문학 야구 경기·관중, 대형 행사, 인천 시간별 날씨 (규칙 변경 후 사용 가능해진 자료).

입력 (external/ 폴더, 데이터 담당이 만든 파일의 복사본)
  munhak_games.csv    KBO 경기일정·결과에서 뽑은 문학(SSG 홈) 경기. 실제 결과 행(source가 kbo_result_*)만 사용
  crowd_munhak.csv    KBO 관중 현황의 문학 정규시즌 경기별 관중 수 (2025·2026)
  events_daily.csv    콘서트·축제 (문학경기장·아시아드경기장·송도달빛축제공원·부천종합운동장 인근), 날짜×역 단위
  school_calendar.csv NEIS 학사일정으로 계산한 날짜별 방학 중 학교 비율 (인천 초·중·고)
  weather_hourly.csv  Open-Meteo 재분석 기반 인천 시간별 날씨 (2025-01-01 ~ 2026-10-31, 인천 중심 1지점)

주의: 2026년 7월 이후의 실제 경기 결과·날씨를 씁니다. 2026-11-11 규칙 변경(승하차 실측만 금지)을 전제로 하며,
      이전 규칙(기준일에 알 수 있던 정보만)에서는 이 파일의 특징을 쓰면 안 됩니다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EXT_DIR = Path(__file__).resolve().parent / "external"
MUNHAK_STATION = 3127   # 문학경기장역

GAME_FEATURES = ["mh_game", "mh_cancelled", "mh_postseason", "mh_game_here", "mh_hours_from_start"]
ATTENDANCE_FEATURES = ["mh_att_log", "mh_att_here"]
EVENT_FEATURES = ["ev_today", "ev_is_concert", "ev_day_index", "ev_n_days", "ev_hours_from_start", "ev_city"]
SCHOOL_FEATURES = ["school_vac_share", "school_vac_share_elem", "school_vac_share_mid", "school_vac_share_high",
                   "school_discretionary_share"]
WEATHER_FEATURES = [
    "temp", "temp_feel", "humidity", "precip", "rain", "snowfall", "wind", "wcode",
    "precip_prev1h", "precip_prev3h", "is_raining",
    "day_precip", "day_precip_6_23", "day_precip_am", "day_precip_pm", "day_snow",
    "day_temp_max", "day_temp_min", "day_temp_mean",
    "is_rainy_day", "is_heavy_rain_day", "is_snow_day", "is_heatwave_day", "is_coldwave_day",
]


def load_games() -> pd.DataFrame:
    """날짜별 문학 홈경기: 열렸는지, 개시 시각, 취소 여부, 포스트시즌 여부. 목록에 없는 날은 경기 없음."""
    g = pd.read_csv(EXT_DIR / "munhak_games.csv", dtype=str).fillna("")
    g = g[g["source"].str.startswith("kbo_result")]
    played = g[g["status"] == "played"].sort_values(["game_date", "start_time_kst"]).drop_duplicates("game_date")
    out = pd.DataFrame({
        "date": played["game_date"].to_numpy(),
        "mh_game": 1,
        "mh_start_hour": played["start_time_kst"].str[:2].astype(int) + played["start_time_kst"].str[3:5].astype(int) / 60,
        "mh_postseason": (played["game_type"] == "postseason").astype(int).to_numpy(),
    })
    cancelled = sorted(set(g.loc[g["status"] == "cancelled", "game_date"]) - set(out["date"]))
    out = pd.concat([out, pd.DataFrame({"date": cancelled, "mh_game": 0, "mh_postseason": 0, "mh_cancelled": 1})], ignore_index=True)
    out["mh_cancelled"] = out["mh_cancelled"].fillna(0).astype(int)
    return out


def load_school() -> pd.DataFrame:
    """날짜별 방학 중인 학교 비율 (NEIS 학사일정, 인천 초·중·고)과 재량휴업 비율."""
    s = pd.read_csv(EXT_DIR / "school_calendar.csv")
    return s[["date"] + SCHOOL_FEATURES]


def load_weather() -> pd.DataFrame:
    w = pd.read_csv(EXT_DIR / "weather_hourly.csv")
    return w[["date", "hour"] + WEATHER_FEATURES]


def load_attendance() -> pd.DataFrame:
    """날짜별 문학 정규시즌 관중 수의 log (더블헤더는 합산). 시범·포스트시즌·비경기일은 목록에 없음."""
    c = pd.read_csv(EXT_DIR / "crowd_munhak.csv", dtype=str)
    c["date"] = c["date"].str.replace("/", "-")
    c["n"] = c["attendance"].str.replace(",", "").astype(float)
    c = c.groupby("date", as_index=False)["n"].sum()
    c["mh_att_log"] = np.log1p(c["n"])
    return c[["date", "mh_att_log"]]


def load_events():
    """야구 외 대형 행사(콘서트·축제). (날짜×역 단위 표, 날짜별 인천 전체 행사 수)를 돌려준다."""
    ev = pd.read_csv(EXT_DIR / "events_daily.csv", dtype={"start_time_kst": str})
    t = ev["start_time_kst"].fillna("")
    ev["start_hour"] = np.where(t != "", t.str[:2].replace("", "0").astype(float) + t.str[3:5].replace("", "0").astype(float) / 60, np.nan)
    ev["is_concert"] = (ev["category"] == "concert").astype(int)
    st = (ev[ev["station_id"] > 0].groupby(["date", "station_id"])
          .agg(ev_is_concert=("is_concert", "max"), ev_day_index=("event_day_index", "min"),
               ev_n_days=("event_n_days", "max"), ev_start_hour=("start_hour", "min")).reset_index())
    st["ev_today"] = 1
    city = ev.groupby("date")["event_name"].nunique().rename("ev_city").reset_index()
    return st, city


def add_external(df: pd.DataFrame) -> pd.DataFrame:
    """(date, station_id, hour) 행에 경기·관중·행사·날씨 특징을 붙인다. 행 수와 순서는 그대로."""
    n = len(df)
    out = df.merge(load_games(), on="date", how="left", validate="many_to_one")
    for c in ["mh_game", "mh_cancelled", "mh_postseason"]:
        out[c] = out[c].fillna(0).astype(int)
    here = (out["station_id"] == MUNHAK_STATION) & (out["mh_game"] == 1)
    out["mh_game_here"] = here.astype(int)                                           # 문학경기장역의 경기일
    out["mh_hours_from_start"] = np.where(here, out["hour"] - out["mh_start_hour"], np.nan)   # 경기 개시 기준 시각
    out = out.drop(columns="mh_start_hour")
    # 관중 수: 날짜 단위 값과, 문학경기장역에만 켜지는 값
    out = out.merge(load_attendance(), on="date", how="left", validate="many_to_one")
    out["mh_att_here"] = np.where(here, out["mh_att_log"], np.nan)
    # 행사: 행사장 역의 행사일에만 켜짐
    st, city = load_events()
    out = out.merge(st, on=["date", "station_id"], how="left", validate="many_to_one")
    out = out.merge(city, on="date", how="left", validate="many_to_one")
    for c in ["ev_today", "ev_is_concert", "ev_city"]:
        out[c] = out[c].fillna(0).astype(int)
    out["ev_hours_from_start"] = out["hour"] - out["ev_start_hour"]                  # 시작 시각을 모르면 NaN
    out = out.drop(columns="ev_start_hour")
    out = out.merge(load_school(), on="date", how="left", validate="many_to_one")
    out = out.merge(load_weather(), on=["date", "hour"], how="left", validate="many_to_one")
    assert len(out) == n and out[WEATHER_FEATURES + SCHOOL_FEATURES].notna().all().all()
    return out
