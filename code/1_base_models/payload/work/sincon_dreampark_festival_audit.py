"""Verify Dream Park visitor egress on the exact pre-event forecast, then stage keyed offsets."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sincon_station_state import KEY
from sincon_weekly_variability import protected_days

SOURCE = {
    "2025_organizer": "https://dreamparkcf.com/sub/board.php?bn=hana_board_16&fn=setting&gb=hana_board_16&mn=business&zest_bn=hana_board_15",
    "2026_official_calendar": "https://korean.visitkorea.or.kr/kfes/detail/fstvlDetail.do?cmsCntntsId=3377402",
    "2026_organizer": "https://www.dreampark.or.kr/main.do",
    "2026_interruption": "https://trip.beomtachi.com/dream-park-chrysanthemum-festival",
    "transport_organizer": "https://slc.or.kr/slc/sli/di/direct01.do",
    "route_87": "https://bustrain.kr/citybus/581",
}
CONTROL_STATIONS = (3203, 3204, 3205, 3208, 3210)
STATIONS = (3206, 3207)
FESTIVAL_2025 = pd.date_range("2025-10-23", "2025-11-05")
FINAL_DAYS = pd.DatetimeIndex(["2026-10-16", "2026-10-17", "2026-10-20", "2026-10-21",
                               "2026-10-22", "2026-10-23", "2026-10-24", "2026-10-25"])
HOURS = tuple(range(13, 19))
BASE_CHECKSUM = "2d836ed13e1a47be96575d16e5abe843c301407e3ef6302422ae049e837741da"
SHRINKAGE = .5
CAP = .25


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def context(work):
    with ZipFile(work / "sincon_local_repro.zip") as archive:
        root = "external_inputs/external/"
        weather = pd.read_csv(archive.open(root + "weather_hourly.csv"), parse_dates=["date"],
                              usecols=["date", "weather_zone", "hour", "precip"])
        regions = pd.read_csv(archive.open(root + "weather_zone_map.csv")).set_index("station_id").weather_zone
    return weather, regions


def ordinary(frame, protected):
    holidays, events = protected
    return ~frame.date.isin(holidays) & ~pd.MultiIndex.from_frame(frame[KEY[:2]]).isin(events)


def episodes(replay, official, protected):
    selected = replay.loc[replay.label.eq("2025-09-30") & replay.date.between("2025-10-01", "2025-11-12")].copy()
    if selected.duplicated(KEY).any() or not np.allclose(
            official.reindex(pd.MultiIndex.from_frame(selected[KEY])), selected.y_log, rtol=0., atol=1e-8):
        raise ValueError("The one fixed-origin forecast must use only unique official historical labels.")
    if selected.date.min() <= pd.Timestamp("2025-09-30") or not selected.date.max() < pd.Timestamp("2026-07-01"):
        raise ValueError("The replay must be entirely within the training period.")
    selected["error"] = selected.y_log - selected.baseline
    controls = selected.loc[selected.station_id.isin(CONTROL_STATIONS) & ordinary(selected, protected)]
    hourly = controls.groupby(["date", "hour"]).error.agg(["median", "size"])
    local = selected.loc[selected.station_id.isin(STATIONS)].join(hourly, on=["date", "hour"])
    local["relative"] = local.error - local["median"].where(local["size"].ge(3))
    local["weekday"] = local.date.dt.day_of_week
    local["weekend"] = local.weekday.ge(5)
    local["active"] = ordinary(local, protected)
    return local


def historical(day_frame, station):
    data = day_frame.loc[day_frame.station_id.eq(station)]
    baseline = data.loc[data.date.lt(FESTIVAL_2025.min()) & data.active].groupby(["weekday", "hour"]).relative.median()
    event = data.loc[data.date.isin(FESTIVAL_2025)].copy()
    if (event.date.nunique() != 14 or event.relative.isna().any() or not event.active.all()
            or not event.groupby("date").hour.size().eq(18).all()):
        raise ValueError("Each analogous event date needs a complete, independent archive.")
    event["contrast"] = event.relative - baseline.reindex(pd.MultiIndex.from_frame(event[["weekday", "hour"]])).to_numpy()
    if event.contrast.isna().any():
        raise ValueError("Every event hour needs an earlier ordinary weekday control.")
    donor = event.loc[event.date.lt("2025-10-30")]
    held = event.loc[event.date.ge("2025-10-30")]
    if donor.date.max() >= held.date.min():
        raise ValueError("The first event week must precede the second event week.")
    hist = {}
    for weekend in (False, True):
        first = donor.loc[donor.weekend.eq(weekend) & donor.hour.isin(HOURS)]
        second = held.loc[held.weekend.eq(weekend) & held.hour.isin(HOURS)]
        reference = first.groupby("hour")[["error", "contrast"]].median()
        log_delta = SHRINKAGE * reference.min(axis=1).clip(0., CAP / SHRINKAGE)
        values = second.hour.map(log_delta).to_numpy()
        improved = {"reference": float(np.sqrt(np.mean(np.square(second.error)))),
                    "candidate": float(np.sqrt(np.mean(np.square(second.error - values))))}
        day_score = {}
        for date, block in second.groupby("date"):
            correction = block.hour.map(log_delta)
            first_error = float(np.sqrt(np.mean(np.square(block.error))))
            updated = float(np.sqrt(np.mean(np.square(block.error - correction))))
            day_score[str(date.date())] = {"rmsle": [first_error, updated], "mean_log_residual": float(block.error.mean())}
        by_day = []
        group = event.loc[event.weekend.eq(weekend) & event.hour.isin(HOURS)].copy()
        for date, block in group.groupby("date"):
            by_day.append({"date": str(date.date()), "observed_count": int(round(np.expm1(block.y_log).sum())),
                           "forecast_count": float(np.expm1(block.baseline).sum()), "mean_residual": float(block.error.mean()),
                           "spatial_weekday_contrast": float(block.contrast.mean())})
        hist["weekend" if weekend else "weekday"] = {
            "out_of_time_rmsle": improved, "out_of_time_by_day": day_score,
            "fit_dates": sorted(first.date.dt.strftime("%Y-%m-%d").unique()),
            "hourly_offsets": {int(hour): float(value) for hour, value in log_delta.items()},
            "historical_station_days": by_day,
            "early_positive_in_late_day": int(sum(r["rmsle"][1] < r["rmsle"][0] for r in day_score.values())),
        }
    return hist


def final_scope(args, report, protected):
    grid = pd.read_csv(args.team / "data/test.csv", parse_dates=["date"], usecols=["example_id", *KEY])
    baseline = pd.read_csv(args.baseline, float_precision="round_trip")
    original = pd.read_csv(args.original, float_precision="round_trip")
    if (digest(args.baseline) != BASE_CHECKSUM or len(grid) != 157_194 or grid.duplicated(KEY).any()
            or not grid.example_id.is_unique or not grid.example_id.equals(baseline.example_id)
            or not grid.example_id.equals(original.example_id)):
        raise ValueError("The current 0.14774-based incumbent must match official test keys.")
    local = grid.loc[grid.station_id.isin(STATIONS)].assign(base=baseline.boardings, original=original.boardings)
    selected = local.loc[local.date.isin(FINAL_DAYS) & local.hour.isin(HOURS)].copy()
    selected["day_type"] = np.where(selected.date.dt.day_of_week.ge(5), "weekend", "weekday")
    # Both open 2025 weekends and the later weekday block supported the direct
    # interchange station. The secondary transfer stop is separately audited.
    choices = {(3207, "weekday"), (3207, "weekend")}
    shifts = {}
    for key in choices:
        check = report[str(key[0])][key[1]]
        scores = check["out_of_time_rmsle"]
        if scores["candidate"] >= scores["reference"]:
            raise ValueError(f"No chronological evidence for the preselected {key} scope.")
        shifts.update({(*key, hour): delta for hour, delta in check["hourly_offsets"].items()})
    selected["log_delta"] = [shifts.get(key, 0.) for key in zip(selected.station_id, selected.day_type, selected.hour)]
    selected = selected.loc[selected.log_delta.gt(0.)]
    if not ordinary(selected, protected).all() or selected.date.eq("2026-10-18").any() or not selected.station_id.eq(3207).all():
        raise ValueError("The event scope must avoid 10/18 and already-protected events.")
    values = np.expm1(np.log1p(selected.base) + selected.log_delta)
    if not np.isfinite(values).all() or np.any(values < 0.) or selected.log_delta.max() > CAP:
        raise ValueError("All staged offsets must give finite valid forecasts within the effect bound.")
    local = local.loc[local.station_id.eq(3207) & local.hour.isin(HOURS)].copy()
    weather, regions = context(args.work_dir)
    rain = weather.loc[weather.weather_zone.eq(regions[3207]) & weather.hour.isin(HOURS)].groupby("date").precip.sum()
    local["date_rain"] = local.date.map(rain)
    comparisons = {}
    for date, rows in selected.groupby("date"):
        others = local.loc[local.date.dt.day_of_week.eq(date.day_of_week) & ~local.date.isin(pd.date_range(
            "2026-10-16", "2026-10-25")) & local.date.between("2026-10-01", "2026-10-31") & local.date_rain.le(3)
                           & ordinary(local, protected)]
        profile = others.groupby("hour").base.median().reindex(rows.hour)
        updated = values.loc[rows.index]
        comparisons[str(date.date())] = {"rows": len(rows), "incumbent": float(rows.base.sum()),
                                         "ordinary_same_weekday_incumbent": float(profile.sum()),
                                         "ordinary_same_weekday_support": int(others.date.nunique()),
                                         "prior_model_log_shift": float((np.log1p(rows.base) - np.log1p(rows.original)).mean()),
                                         "candidate": float(updated.sum())}
    selected[["example_id", *KEY, "log_delta"]].to_csv(args.delta, index=False, float_format="%.12f")
    return {"incumbent_sha256": digest(args.baseline), "official_test_sha256": digest(args.team / "data/test.csv"),
            "delta_sha256": digest(args.delta), "changed_rows": len(selected),
            "applied_days": comparisons, "delta_min_max": [float(selected.log_delta.min()), float(selected.log_delta.max())],
            "forecast_total_before": float(selected.base.sum()), "forecast_total_after": float(values.sum()),
            "excluded_operational_dates": ["2026-10-18 (weather interruption)", "2026-10-19 (phased resumption)"],
            "source_event_count": 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "archive", "baseline", "original", "delta", "results"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    truth = pd.read_csv(args.team / "data/train.csv", parse_dates=["date"], usecols=[*KEY, "boardings"])
    official = np.log1p(truth.set_index(KEY).boardings)
    replay = pd.read_pickle(args.archive)
    protected = protected_days(args.work_dir)
    prior = episodes(replay, official, protected)
    history = {str(station): historical(prior, station) for station in STATIONS}
    final = final_scope(args, history, protected)
    result = {"sources": SOURCE, "historical_exact_15126": history, "final": final}
    args.results.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
