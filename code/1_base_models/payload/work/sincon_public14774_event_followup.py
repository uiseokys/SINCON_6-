"""Replay and apply the science festival departure and Incheon trail event corrections."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_public14774_event_update import digest, submission
from sincon_songdo_schedule import save_submission
from sincon_station_state import KEY


BASELINE_SHA256 = "880628c6a5bec36a8fd0b3262c230a5eb9575cd10ff83ba415ebb85d097b7fde"
ORIGINS = ("summer", "2025-07-31", "2025-08-31", "2025-09-30")
STATIONS = (3209, 3210)
HOURS = list(range(14, 19))
OLD_WEIGHTS = np.array([0., .5, .5, .5, 0.])
PEAK_WEIGHTS = np.array([0., .5, .75, .75, 0.])
SHOULDER_WEIGHTS = np.array([.25, .5, .75, .75, .25])
DATE_PAIRS = tuple(zip(pd.to_datetime(["2025-10-24", "2025-10-25"]), pd.to_datetime(["2026-10-23", "2026-10-24"])))
TRAIL_IDS = {"J156966", "J156967"}


def station_profiles(rows, station, values):
    station_hours = rows.loc[rows.station_id.eq(station) & rows.hour.isin(HOURS)]
    return station_hours.pivot(index="date", columns="hour", values=values).reindex(columns=HOURS)


def earlier_controls(source, station, date):
    dates = pd.to_datetime(source["historical"][f"{station}:{date.date()}"]["control_dates"])
    if len(dates) < 3 or not (dates < date).all() or not (dates.dayofweek == date.dayofweek).all():
        raise ValueError("The science comparison requires earlier same-weekday controls.")
    return dates


def excess(history, source, station, date):
    controls = earlier_controls(source, station, date)
    result = (history.loc[date] - history.loc[controls].median()).clip(lower=0.)
    if not np.isfinite(result).all():
        raise ValueError("Every transferred hour must use official observations and complete controls.")
    return result


def rmsle(truth, prediction):
    return float(np.sqrt(np.mean(np.square(np.log1p(truth) - np.log1p(prediction)))))


def science_backtest(history, replay, source):
    first, second = (day[0] for day in DATE_PAIRS)
    report = {}
    for label in ORIGINS:
        rows = replay.loc[replay.label.eq(label) & replay.station_id.isin(STATIONS) & replay.hour.isin(HOURS)]
        rows = rows.merge(history, on=KEY, how="left", validate="one_to_one")
        if not np.allclose(rows.y_log, np.log1p(rows.boardings), atol=1e-9, rtol=0.):
            raise ValueError(f"Archived outcomes must match official training rows in {label}.")
        measurements = []
        for station in STATIONS:
            actual = station_profiles(history, station, "boardings")
            predictions = np.expm1(station_profiles(rows, station, "baseline"))
            controls = predictions.index.intersection(earlier_controls(source, station, second))
            normal = predictions.loc[controls].median()
            demand = excess(actual, source, station, first)
            baseline = predictions.loc[second]
            old = np.maximum(baseline, normal + demand * OLD_WEIGHTS)
            peak = np.maximum(old, normal + demand * PEAK_WEIGHTS)
            full = np.maximum(peak, normal + demand * (SHOULDER_WEIGHTS if station == 3210 else PEAK_WEIGHTS))
            measurements.append(pd.DataFrame({"truth": actual.loc[second], "old": old, "peak": peak, "candidate": full,
                                              "station": station, "control_dates": len(controls)}))
        result = pd.concat(measurements)
        details = {name: rmsle(result.truth, result[name]) for name in ("old", "peak", "candidate")}
        if not details["candidate"] < details["peak"] < details["old"]:
            raise ValueError(f"The departure and shoulder corrections must improve the second day in {label}.")
        report[label] = {"rmsle": details, "rows": len(result), "control_dates": int(result.control_dates.min())}
    return report


def science_delta(grid, history, source):
    results = []
    for station in STATIONS:
        actual = station_profiles(history, station, "boardings")
        predictions = station_profiles(grid, station, "base")
        for prior, date in DATE_PAIRS:
            plan = source["unvalidated_projection_only"][f"{station}:{date.date()}"]
            if plan["source"] != str(prior.date()):
                raise ValueError("Each 2026 event day must match the verified 2025 weekday.")
            controls = pd.to_datetime(plan["unlabelled_control_dates"])
            if len(controls) < 3 or not (controls < date).all() or not (controls.dayofweek == date.dayofweek).all():
                raise ValueError("Final ordinary prediction anchors must predate the event and match its weekday.")
            normal = predictions.loc[controls].median()
            weights = {16: .75, 17: .75}
            if station == 3210:
                weights.update({14: .25, 18: .25})
            change = pd.Series(weights) * excess(actual, source, station, prior).reindex(weights)
            floors = normal.reindex(weights) + change
            target = grid.loc[grid.station_id.eq(station) & grid.date.eq(date) & grid.hour.isin(weights)].copy()
            if len(target) != len(weights) or not np.isfinite(floors).all():
                raise ValueError("Science adjustments must match the bounded event schedule.")
            target["after"] = np.maximum(target.base, target.hour.map(floors))
            results.append(target.assign(event="science_festival"))
    result = pd.concat(results)
    if len(result) != 12 or not result.after.gt(result.base).all():
        raise ValueError("The science follow-up must adjust exactly the twelve supported hours.")
    return result


def trail_delta(path, grid):
    change = pd.read_csv(path, usecols=["example_id", *KEY, "log_delta"], parse_dates=["date"], float_precision="round_trip")
    if (set(change.example_id) != TRAIL_IDS or len(change) != 2 or not change.date.eq("2026-10-31").all()
            or not change.station_id.eq(3226).all() or set(change.hour) != {12, 13}
            or not change.log_delta.between(0., .10, inclusive="right").all()):
        raise ValueError("Trail event offsets must match the two supported station-hours.")
    target = grid.merge(change, on=["example_id", *KEY], validate="one_to_one")
    if len(target) != 2:
        raise ValueError("The trail event offsets must match official test IDs and station-hours.")
    return target.assign(after=np.expm1(np.log1p(target.base) + target.log_delta), event="trail_festival")


def run(args):
    official = pd.read_csv(args.team / "data/train.csv", usecols=[*KEY, "boardings"], parse_dates=["date"])
    if official.date.max() != pd.Timestamp("2026-06-30") or official.duplicated(KEY).any():
        raise ValueError("Only the official historical target observations may be used.")
    history = official.loc[official.station_id.isin(STATIONS) & official.hour.isin(HOURS)]
    source = json.loads(args.science_source.read_text())
    historical = science_backtest(history, pd.read_pickle(args.archive), source)
    if digest(args.baseline) != BASELINE_SHA256:
        raise ValueError("Start from the verified festival candidate.")
    grid = pd.read_csv(args.team / "data/test.csv", usecols=["example_id", *KEY], parse_dates=["date"])
    if len(grid) != 157_194 or not grid.example_id.is_unique or grid.duplicated(KEY).any():
        raise ValueError("Use the complete official test grid.")
    baseline = submission(args.baseline, grid)
    grid["base"] = baseline.boardings
    adjustments = [science_delta(grid, history, source)]
    if args.trail:
        adjustments.append(trail_delta(args.trail, grid))
    changes = pd.concat(adjustments).set_index("example_id", verify_integrity=True)
    if not np.isfinite(changes.after).all() or not changes.after.gt(changes.base).all():
        raise ValueError("All event adjustments must be finite and positive.")
    changed = grid.example_id.isin(changes.index)
    values = grid.base.to_numpy(copy=True)
    values[changed] = changes.after.reindex(grid.example_id[changed])
    if not np.array_equal(values != grid.base, changed):
        raise ValueError("Only the recorded event rows may change.")
    save_submission(args.output, baseline, grid, values)
    report = {"historical_first_day": str(DATE_PAIRS[0][0].date()), "historical_second_day": str(DATE_PAIRS[1][0].date()),
              "science_backtest": historical, "submission": str(args.output.resolve()), "sha256": digest(args.output),
              "baseline_sha256": BASELINE_SHA256, "test_rows": len(grid),
              "changes_by_event": changes.groupby("event").size().to_dict(),
              "changes": json.loads(changes.reset_index().to_json(orient="records", date_format="iso", double_precision=15))}
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["changes_by_event"], report["sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "archive", "science_source", "baseline", "output", "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--trail", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
