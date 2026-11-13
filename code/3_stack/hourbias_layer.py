"""일반일 역·요일·시간대 오차 보정 (K1) 과 휴일·행사 행 묶음 (K3) 계산 함수.

모든 상수는 2026-06-30 이전의 과거 예측과 공식 train.csv 실측만으로 정했고,
Public 점수를 본 뒤 바꾸지 않았다 (CANDIDATE_SET_PREREGISTRATION.md).
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["date", "station_id", "hour"]
DAYKEY = ["date", "station_id"]
MODEL = "lunar3122"            # 주후보 최종 단계의 과거 예측 열 이름
FIT = ["station_id", "dow", "hour"]
ORIGIN = pd.Timestamp("2026-06-30")
WINDOW_DAYS = 84               # 기준일 직전 84일의 과거 오차만 쓴다
SHRINK = 12                    # 수축: n / (n + 12)
SCALE = 0.25                   # 배율
MIN_DAYS = 6                   # 역·요일·시간대별 관측 6일 이상일 때만
EXCLUDED_STATIONS = {3107, 3108, 3109, 3127, 3139, 3209, 3754}
CALENDAR_FLAGS = ["is_holiday", "is_lunar_season", "is_bridge_day", "is_quasi_holiday"]
TOLERANCE = 1e-9


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2 ** 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_protection(cache):
    """보정에서 빼는 날짜와 역·날짜 목록을 읽는다."""
    cache = Path(cache)
    cal = pd.read_csv(cache / "calendar_flags.csv", parse_dates=["date"])
    days = cal.loc[cal[CALENDAR_FLAGS].ne(0).any(axis=1), "date"]
    events = pd.read_csv(cache / "events_frozen_v3.csv", parse_dates=["date"])
    protected = pd.read_csv(cache / "protected_event_keys.csv", parse_dates=["date"])
    return days, pd.MultiIndex.from_frame(events[DAYKEY]), pd.MultiIndex.from_frame(protected[DAYKEY].drop_duplicates())


def quiet_rows(frame, days, events, extra=None):
    """공휴일·명절 기간·징검다리·준공휴일, 제외 역, 행사 역·날짜가 아닌 '일반' 행."""
    q = (~frame["date"].isin(days) & ~frame["station_id"].isin(EXCLUDED_STATIONS)
         & ~pd.MultiIndex.from_frame(frame[DAYKEY]).isin(events))
    if extra is not None:
        q &= ~pd.MultiIndex.from_frame(frame[DAYKEY]).isin(extra)
    return q


def select_history(rows, origin=ORIGIN):
    """기준일 직전 84일, 기준일보다 앞선 시점에 만든 예측만. 겹치면 가장 늦은 예측 하나."""
    m = (rows["date"].gt(origin - pd.Timedelta(days=WINDOW_DAYS)) & rows["date"].le(origin)
         & rows["origin"].lt(origin) & rows["quiet"])
    return rows.loc[m].sort_values("origin").drop_duplicates(KEY, keep="last")


def fit_stats(history):
    """역·요일·시간대별 평균 오차(역·날짜 평균을 뺀 값)에 수축과 배율을 적용한다."""
    history = history.copy()
    history["error"] = history["y_log"] - history[MODEL]
    g = history.groupby(DAYKEY)["error"]
    history["error"] = (history["error"] - g.transform("mean")).where(g.transform("size").eq(18))
    stats = history.groupby(FIT)["error"].agg(["mean", "count"])
    stats["correction"] = (stats["mean"] * SCALE * stats["count"] / (stats["count"] + SHRINK)).where(stats["count"].ge(MIN_DAYS))
    return stats


def apply_stats(target, stats, is_quiet):
    """18개 시간대 보정값이 모두 있는 일반 역·날짜에만, 역·날짜 평균이 0이 되게 적용한다."""
    t = target.join(stats[["correction"]], on=FIT)
    good = is_quiet & t.groupby(DAYKEY)["correction"].transform("count").eq(18)
    delta = (t["correction"] - t.groupby(DAYKEY)["correction"].transform("mean")).where(good, 0.0)
    return delta.to_numpy(), good.to_numpy()


def same_predictions(a, b, tolerance=TOLERANCE):
    """두 제출 파일의 ID·순서가 같고 log1p 예측이 허용 오차 안에서 같은지."""
    if len(a) != len(b) or not (a["example_id"].to_numpy() == b["example_id"].to_numpy()).all():
        return False, float("inf")
    d = np.abs(np.log1p(a["boardings"].to_numpy()) - np.log1p(b["boardings"].to_numpy()))
    return bool(d.max() <= tolerance), float(d.max())


def build_k1(test, main, history_rows, cache):
    """K1: 주후보 + 일반일 역·요일·시간대 오차 보정."""
    days, events, protected = read_protection(cache)
    rows = history_rows.copy()
    rows["dow"] = rows["date"].dt.day_of_week
    rows["quiet"] = quiet_rows(rows, days, events)
    history = select_history(rows)
    stats = fit_stats(history)
    frame = test[["example_id", *KEY]].copy()
    frame["dow"] = frame["date"].dt.day_of_week
    frame[MODEL] = np.log1p(main["boardings"].to_numpy())
    is_quiet = quiet_rows(frame, days, events, protected)
    delta, good = apply_stats(frame, stats, is_quiet)
    base = frame[MODEL].to_numpy()
    new = base + delta
    day_mean = pd.Series(delta).groupby([frame["date"].to_numpy(), frame["station_id"].to_numpy()]).mean().abs().max()
    if day_mean > 1e-12 or (new < 0).any() or not np.array_equal(new[~good], base[~good]):
        raise ValueError("보정은 역·날짜 log 평균과 보호 행을 그대로 두어야 한다.")
    out = main.copy()
    out.loc[good, "boardings"] = np.expm1(new[good])
    info = {"history_rows": int(len(history)), "history_first_date": str(history["date"].min().date()),
            "history_last_date": str(history["date"].max().date()),
            "history_origins": sorted(str(o.date()) for o in history["origin"].unique()),
            "cells_with_correction": int(stats["correction"].notna().sum()),
            "changed_rows": int(good.sum()), "changed_station_days": int(good.sum() // 18),
            "max_abs_log_change": float(np.abs(delta).max())}
    return out, info


def build_k3(k1, main, monday, cup_walk):
    """K3: K1 위에, 월요일 대체공휴일 후보와 코리아컵·걷기대회 후보가 주후보와 다른 행을 그대로 옮긴다."""
    base = np.log1p(main["boardings"].to_numpy())
    mon_rows = np.abs(np.log1p(monday["boardings"].to_numpy()) - base) > 1e-12
    cup_rows = np.abs(np.log1p(cup_walk["boardings"].to_numpy()) - base) > 1e-12
    if (mon_rows & cup_rows).any():
        raise ValueError("두 보정이 같은 행을 바꾸면 안 된다.")
    out = k1.copy()
    out.loc[mon_rows, "boardings"] = monday["boardings"].to_numpy()[mon_rows]
    out.loc[cup_rows, "boardings"] = cup_walk["boardings"].to_numpy()[cup_rows]
    return out, {"monday_rows": int(mon_rows.sum()), "event_rows": int(cup_rows.sum())}


def check_submission(frame, test):
    v = frame["boardings"].to_numpy()
    ok = (list(frame.columns) == ["example_id", "boardings"] and len(frame) == 157194
          and (frame["example_id"].to_numpy() == test["example_id"].to_numpy()).all()
          and np.isfinite(v).all() and (v >= 0).all())
    if not ok:
        raise ValueError("제출 형식 검사 실패")
    return True
