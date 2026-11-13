"""달력 특징 생성 (OT 자료 8쪽 '달력 정보'): 2025-01-01 ~ 2026-10-31, 하루 1행.

실행: python make_calendar.py   ->  calendar_features.csv

여기 들어가는 정보는 모두 예측 기준일(2026-06-30) 전에 정해져 있던 일정입니다.
  - 공휴일: 「관공서의 공휴일에 관한 규정」의 법정공휴일, 대체공휴일, 선거일, 2025-01-27 임시공휴일.
            2026년 노동절(5/1)·제헌절(7/17)은 2026년 상반기에 공휴일로 확정된 날입니다.
  - 방학: 학교별 실제 일정이 아니라 통상 학사일정에 따른 근사 구간입니다(아래 상수).
          외부 조회 자료를 쓰지 않았으므로 필요하면 이 구간만 고치면 됩니다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

START, END = "2025-01-01", "2026-10-31"

# (날짜, 이름, 종류)  종류: 1=일반 공휴일, 2=설·추석 연휴 3일, 3=대체·임시공휴일·선거일
HOLIDAYS = [
    ("2025-01-01", "신정", 1), ("2025-01-27", "임시공휴일", 3),
    ("2025-01-28", "설 연휴", 2), ("2025-01-29", "설날", 2), ("2025-01-30", "설 연휴", 2),
    ("2025-03-01", "삼일절", 1), ("2025-03-03", "대체공휴일", 3),
    ("2025-05-05", "어린이날·부처님오신날", 1), ("2025-05-06", "대체공휴일", 3),
    ("2025-06-03", "대통령선거", 3), ("2025-06-06", "현충일", 1), ("2025-08-15", "광복절", 1),
    ("2025-10-03", "개천절", 1),
    ("2025-10-05", "추석 연휴", 2), ("2025-10-06", "추석", 2), ("2025-10-07", "추석 연휴", 2),
    ("2025-10-08", "대체공휴일", 3), ("2025-10-09", "한글날", 1), ("2025-12-25", "성탄절", 1),
    ("2026-01-01", "신정", 1),
    ("2026-02-16", "설 연휴", 2), ("2026-02-17", "설날", 2), ("2026-02-18", "설 연휴", 2),
    ("2026-03-01", "삼일절", 1), ("2026-03-02", "대체공휴일", 3),
    ("2026-05-01", "노동절", 1), ("2026-05-05", "어린이날", 1),
    ("2026-05-24", "부처님오신날", 1), ("2026-05-25", "대체공휴일", 3),
    ("2026-06-03", "지방선거", 3), ("2026-06-06", "현충일", 1),
    ("2026-07-17", "제헌절", 1), ("2026-08-15", "광복절", 1), ("2026-08-17", "대체공휴일", 3),
    ("2026-09-24", "추석 연휴", 2), ("2026-09-25", "추석", 2), ("2026-09-26", "추석 연휴", 2),
    ("2026-10-03", "개천절", 1), ("2026-10-05", "대체공휴일", 3), ("2026-10-09", "한글날", 1),
]
LUNAR_DAY = ["2025-01-29", "2025-10-06", "2026-02-17", "2026-09-25"]   # 설날·추석 당일
# 공휴일은 아니지만 쉬는 사람이 많은 날 (2025년 근로자의 날은 관공서 공휴일이 아니었음)
SEMI_HOLIDAYS = ["2025-05-01"]

# 방학 근사 구간 (통상 학사일정). 초·중·고: 7월 하순~8월 중순, 1월 초~2월 말 / 대학: 종강 후~개강 전
SCHOOL_VACATION = [("2025-01-01", "2025-03-03"), ("2025-07-21", "2025-08-17"),
                   ("2026-01-05", "2026-03-02"), ("2026-07-21", "2026-08-17")]
UNIV_VACATION = [("2025-01-01", "2025-03-03"), ("2025-06-21", "2025-08-31"),
                 ("2025-12-20", "2026-03-02"), ("2026-06-20", "2026-08-31")]
# 직장인 여름휴가 성수기 (매년 7월 마지막 주 ~ 8월 첫 주)
SUMMER_PEAK = [("2025-07-26", "2025-08-10"), ("2026-07-25", "2026-08-09")]


def _in_ranges(dates: pd.Series, ranges) -> np.ndarray:
    mask = np.zeros(len(dates), dtype=bool)
    for s, e in ranges:
        mask |= ((dates >= s) & (dates <= e)).to_numpy()
    return mask.astype(int)


def _blocks(off: np.ndarray):
    """연속 휴무 구간의 길이와 구간 안 순번(1부터). 근무일은 0."""
    length = np.zeros(len(off), dtype=int)
    pos = np.zeros(len(off), dtype=int)
    i = 0
    while i < len(off):
        if off[i]:
            j = i
            while j < len(off) and off[j]:
                j += 1
            length[i:j] = j - i
            pos[i:j] = np.arange(1, j - i + 1)
            i = j
        else:
            i += 1
    return length, pos


def _distance(flag: np.ndarray, forward: bool) -> np.ndarray:
    """가장 가까운 True까지의 일수 (forward=True면 앞으로, False면 뒤로). 자신이 True면 0, 최대 14."""
    out = np.full(len(flag), 14, dtype=int)
    last = None
    order = range(len(flag) - 1, -1, -1) if forward else range(len(flag))
    for i in order:
        if flag[i]:
            last = i
        if last is not None:
            out[i] = min(abs(last - i), 14)
    return out


def build_calendar() -> pd.DataFrame:
    # 기간 양끝에서도 연휴 길이·거리가 맞도록 앞뒤로 여유를 두고 계산한 뒤 잘라낸다.
    d = pd.Series(pd.date_range("2024-12-01", "2026-11-30", freq="D"))
    ds = d.dt.strftime("%Y-%m-%d")
    cal = pd.DataFrame({"date": ds})
    kind = dict((h[0], h[2]) for h in HOLIDAYS)
    kind["2024-12-25"] = 1   # 여유 구간의 성탄절

    cal["year"] = d.dt.year
    cal["month"] = d.dt.month
    cal["dow"] = d.dt.day_of_week                              # 0=월 ... 6=일
    cal["is_weekend"] = (cal["dow"] >= 5).astype(int)
    cal["season"] = cal["month"].map({12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3})
    cal["days_since_start"] = (d - pd.Timestamp(START)).dt.days   # 수준 변화(성장·신규역)를 잡는 시간 축

    cal["holiday_kind"] = ds.map(kind).fillna(0).astype(int)   # 0=아님, 1=일반, 2=설·추석, 3=대체·임시·선거
    cal["is_holiday"] = (cal["holiday_kind"] > 0).astype(int)
    cal["is_lunar_day"] = ds.isin(LUNAR_DAY).astype(int)       # 설날·추석 당일
    cal["is_semi_holiday"] = ds.isin(SEMI_HOLIDAYS).astype(int)
    off = ((cal["is_weekend"] == 1) | (cal["is_holiday"] == 1)).to_numpy()
    cal["is_offday"] = off.astype(int)                         # 주말 또는 공휴일
    cal["is_weekday_holiday"] = ((cal["is_weekend"] == 0) & (cal["is_holiday"] == 1)).astype(int)

    block_len, block_pos = _blocks(off)
    cal["offday_block_len"] = block_len                        # 연속 휴무 길이 (근무일 0)
    cal["offday_block_pos"] = block_pos                        # 그 안에서 몇째 날
    # 설·추석이 들어 있는 휴무 구간 전체 (붙은 주말·대체공휴일 포함)
    start_idx = np.where(off, np.arange(len(off)) - block_pos + 1, -1)
    lunar_starts = set(start_idx[(cal["holiday_kind"] == 2).to_numpy()])
    cal["is_lunar_block"] = np.isin(start_idx, list(lunar_starts)).astype(int)

    prev_off = np.r_[False, off[:-1]]
    next_off = np.r_[off[1:], False]
    cal["is_bridge_day"] = (~off & prev_off & next_off).astype(int)                       # 징검다리 근무일
    cal["is_before_long_off"] = (~off & (np.r_[block_len[1:], 0] >= 3)).astype(int)       # 3일 이상 연휴 전날
    cal["is_after_long_off"] = (~off & (np.r_[0, block_len[:-1]] >= 3)).astype(int)       # 3일 이상 연휴 다음날
    cal["days_to_offday"] = _distance(off, forward=True)
    cal["days_from_offday"] = _distance(off, forward=False)

    cal["is_school_vacation"] = _in_ranges(ds, SCHOOL_VACATION)
    cal["is_univ_vacation"] = _in_ranges(ds, UNIV_VACATION)
    cal["is_summer_peak"] = _in_ranges(ds, SUMMER_PEAK)
    # 설날·추석 당일 기준 며칠 전후인지 (-5 ~ +5, 그 밖은 NaN). 연휴 길이가 해마다 달라도 같은 기준으로 비교됨
    rel = np.full(len(d), np.nan)
    for lunar in pd.to_datetime(pd.Series(LUNAR_DAY)):
        k = (d - lunar).dt.days.to_numpy()
        near = np.abs(k) <= 5
        rel[near] = k[near]
    cal["lunar_rel_day"] = rel

    return cal[(ds >= START) & (ds <= END)].reset_index(drop=True)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "calendar_features.csv"
    cal = build_calendar()
    cal.to_csv(out, index=False)
    print(f"저장: {out.name} {cal.shape}, 공휴일 {int(cal['is_holiday'].sum())}일")
