"""Verify a bounded 21:00 Seonhak festival increase using historical donor rows."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_dreampark_festival_audit import context
from sincon_weekly_variability import protected_days

KEY = ["date", "station_id", "hour"]
BASE_SHA256 = "2d836ed13e1a47be96575d16e5abe843c301407e3ef6302422ae049e837741da"
FORECAST_DATES = pd.to_datetime(["2026-10-09", "2026-10-10"])
DONOR_DATES = pd.to_datetime(["2025-10-18", "2025-10-19"])
CAP_COUNTS = 43.75
CAP_LOG = .25
SOURCES = {
    "2026_program": "https://www.ysfac.or.kr/user/cultural/events_view.php?search=&sq=400",
    "2026_station_access": "https://www.yeonsu.go.kr/tour/festival/neungheodae_festival/location.asp",
    "2025_festival": "https://www.yeonsu.go.kr/etc/bbs/SearchView.asp?bbs_code=board_3&seq=28296",
    "2025_seonhak_opening": "https://www.yeonsu.go.kr/culture/show/art_show.asp?idx=827&page=v",
    "2025_munhak_jin_both_nights": "https://weverse.io/bts/notice/30499?hl=ko",
}


def audit_history(team, work):
    historical = pd.read_csv(team / "data/train.csv", parse_dates=["date"], usecols=[*KEY, "boardings"])
    if historical.date.max() >= FORECAST_DATES.min() or historical.duplicated(KEY).any():
        raise ValueError("Use the unique official historical boardings only.")
    hourly = historical.loc[historical.hour.eq(21)].pivot(index="date", columns="station_id", values="boardings")
    weather, regions = context(work)
    rain = weather.loc[weather.weather_zone.eq(regions[3139]) & weather.hour.between(17, 21)].groupby("date").precip.sum()
    holidays, events = protected_days(work)
    excluded_events = events[events.get_level_values("station_id").isin([3138, 3139])].get_level_values("date")
    gap = hourly[3139] - hourly[3138]
    donors = []
    for day in DONOR_DATES:
        controls = gap.loc[(gap.index.dayofweek == day.dayofweek) & (gap.index < day)
                           & (gap.index >= day - pd.Timedelta(weeks=9))
                           & ~gap.index.isin(holidays.union(excluded_events)) & rain.reindex(gap.index).le(.2)]
        if len(controls) < 4 or rain.loc[day] > .2:
            raise ValueError("Each festival day needs dry earlier same-weekday controls.")
        excess = float(gap.loc[day] - controls.median())
        donors.append({"date": str(day.date()), "boards_3139": int(hourly.loc[day, 3139]),
                       "boards_3138": int(hourly.loc[day, 3138]), "control_dates": controls.index.strftime("%Y-%m-%d").tolist(),
                       "control_gap_median": float(controls.median()), "net_excess": excess})
    if min(row["net_excess"] for row in donors) < 2 * CAP_COUNTS:
        raise ValueError("Dry same-festival comparisons no longer support the proposed cap.")
    # Munhak hosted the same concert on consecutive nights. The Seonhak park
    # opening took place on the first night only, so the second night is useful
    # as a directional check even though the nights had different weekdays.
    same_venue = []
    for day in pd.to_datetime(["2025-10-31", "2025-11-01"]):
        same_venue.append({"date": str(day.date()), "boards_3128": int(hourly.loc[day, 3128]),
                           "boards_3127": int(hourly.loc[day, 3127])})
    if same_venue[0]["boards_3128"] <= same_venue[1]["boards_3128"] or same_venue[0]["boards_3127"] >= same_venue[1]["boards_3127"]:
        raise ValueError("The paired same-venue check has lost its directional support.")
    return {"dry_h21_neunghudae_donors": donors, "seonhak_park_directional_check": same_venue}


def run(team, baseline_path, delta_path, report_path, work):
    history = audit_history(team, work)
    if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError("Expected the verified combined model and event forecast.")
    grid = pd.read_csv(team / "data/test.csv", usecols=["example_id", *KEY], parse_dates=["date"])
    baseline = pd.read_csv(baseline_path, float_precision="round_trip")
    if len(grid) != 157_194 or not grid.example_id.is_unique or not grid.example_id.equals(baseline.example_id):
        raise ValueError("Baseline must use the complete test grid in order.")
    selected = grid.loc[grid.date.isin(FORECAST_DATES) & grid.station_id.eq(3128) & grid.hour.eq(21)].copy()
    if selected.example_id.tolist() != ["J128193", "J129471"]:
        raise ValueError("Expected the two verified Seonhak station-hours.")
    before = baseline.boardings.loc[selected.index].to_numpy()
    delta = np.minimum(np.log1p(CAP_COUNTS / (before + 1)), CAP_LOG)
    if not np.isfinite(delta).all() or not (delta > 0).all():
        raise ValueError("Expected a finite positive adjustment.")
    selected["log_delta"] = delta
    selected.to_csv(delta_path, index=False, float_format="%.15f")
    changes = selected.assign(before=before, after=np.expm1(np.log1p(before) + delta))
    result = {"sources": SOURCES, "baseline_sha256": BASE_SHA256, "history": history,
              "cap_counts": CAP_COUNTS, "cap_log": CAP_LOG,
              "changes": json.loads(changes.to_json(orient="records", date_format="iso", double_precision=15))}
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "baseline", "delta", "report"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.team, args.baseline, args.delta, args.report, Path(__file__).resolve().parent)
