"""Calibrate ordinary 23:00 predictions near zero from earlier archived errors."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sincon_complementary_blend import historical_station_shift, rmsle
from sincon_source_reweight import evaluate
from sincon_station_state import KEY, WINDOWS
from sincon_toll_directional_local import history_and_baselines
from sincon_weekly_variability import protected_days


COUNT_EDGES = [2., 5., 10.]
BINS = ["0_to_2", "above_2_to_5", "above_5_to_10"]
PRIMARY = 1
MIN_GAIN = .003
DAY = KEY[:2]


def select_ordinary(frame, holidays, events):
    mask = frame.hour.eq(23) & ~frame.date.isin(holidays) & ~pd.MultiIndex.from_frame(frame[DAY]).isin(events)
    eligible = frame.loc[mask, ["date", "station_id", "forecast"]].copy()
    eligible["count_bin"] = np.searchsorted(np.log1p(COUNT_EDGES), eligible.forecast.to_numpy(), side="left")
    eligible["weekday_group"] = np.maximum(eligible.date.dt.dayofweek.to_numpy() - 4, 0)
    return eligible.loc[eligible.count_bin.lt(len(BINS))]


def correction(archive, target, origin, holidays, events):
    if target.date.min() <= origin:
        raise ValueError("Forecast rows must follow the cutoff for fitting residuals.")
    known = archive.loc[archive.origin.lt(origin) & archive.date.le(origin)]
    known = known.sort_values("origin").drop_duplicates(KEY, keep="last").reset_index(drop=True)
    past = select_ordinary(known, holidays, events)
    past["error"] = (known.loc[past.index, "y_log"] - past.forecast.clip(lower=0.)).clip(-.65, .65)

    # One metro-wide mean per date prevents busy dates with many matching
    # stations from dominating the correction. Prior strengths are in days.
    by_date = past.groupby(["date", "count_bin", "weekday_group"]).error.mean().reset_index()
    pooled = by_date.groupby("count_bin").error.agg(["sum", "count"])
    pooled["global"] = (pooled["sum"] / (pooled["count"] + 20.)).where(pooled["count"].ge(12), 0.)
    by_weekday = by_date.groupby(["count_bin", "weekday_group"]).error.agg(["sum", "count"])
    by_weekday = by_weekday.join(pooled["global"], on="count_bin", validate="many_to_one")
    by_weekday["weekday"] = ((by_weekday["sum"] + 15. * by_weekday["global"]) / (by_weekday["count"] + 15.)).where(
        by_weekday["count"].ge(5), by_weekday["global"])
    by_station = past.groupby(["station_id", "count_bin", "weekday_group"]).error.agg(["sum", "count"])
    by_station = by_station.join(by_weekday["weekday"], on=["count_bin", "weekday_group"], validate="many_to_one")
    by_station["local"] = ((by_station["sum"] + 35. * by_station["weekday"]) / (by_station["count"] + 35.)).where(
        by_station["count"].ge(12), by_station["weekday"])

    future = select_ordinary(target.reset_index(drop=True), holidays, events)
    future = future.join(pooled["global"], on="count_bin", validate="many_to_one")
    future = future.join(by_weekday["weekday"], on=["count_bin", "weekday_group"], validate="many_to_one")
    future = future.join(by_station["local"], on=["station_id", "count_bin", "weekday_group"], validate="many_to_one")
    estimates = future.local.fillna(future.weekday).fillna(future["global"]).fillna(0.).clip(-.25, .25)
    all_bins = np.zeros(len(target))
    all_bins[future.index] = estimates
    primary = np.zeros(len(target), dtype=bool)
    primary[future.index[future.count_bin.eq(PRIMARY)]] = True
    counts = pooled["count"].reindex(range(len(BINS)), fill_value=0).astype(int)
    return all_bins, primary, dict(zip(BINS, counts.tolist()))


def uncertainty(dates, actual, before, after, block_days):
    # Resample entire contiguous calendar blocks together with all their rows.
    groups = (dates - dates.min()).dt.days.to_numpy() // block_days
    error = pd.DataFrame({"baseline": (actual - np.maximum(before, 0.)) ** 2,
                          "candidate": (actual - np.maximum(after, 0.)) ** 2, "rows": 1})
    blocks = error.groupby(groups).sum().to_numpy()
    random = np.random.default_rng(0)
    sampled = blocks[random.integers(len(blocks), size=(4000, len(blocks)))].sum(axis=1)
    gain = np.sqrt(sampled[:, 0] / sampled[:, 2]) - np.sqrt(sampled[:, 1] / sampled[:, 2])
    return np.quantile(gain, [.025, .975]).tolist()


def report(frame, archive, origin, holidays, events, station_shift):
    delta, primary, fitted_days = correction(archive, frame, origin, holidays, events)
    actual = frame.y_log.to_numpy()
    before = frame.forecast.to_numpy()
    scores = {}
    for name, baseline in (("raw", before), ("station_calibrated", None if station_shift is None else before + station_shift)):
        if baseline is None:
            continue
        overall = rmsle(actual, baseline)
        scores[name] = {}
        for variant, adjustment in (("primary_2_to_5", np.where(primary, delta, 0.)), ("all_low_count_bins", delta)):
            estimate = baseline + adjustment
            score = rmsle(actual, estimate)
            by_segment = {}
            for segment, selected in (("all_23h", frame.hour.eq(23).to_numpy()), ("primary_2_to_5", primary)):
                if selected.any():
                    by_segment[segment] = {"baseline": rmsle(actual[selected], baseline[selected]),
                                           "corrected": rmsle(actual[selected], estimate[selected])}
            scores[name][variant] = {"baseline": overall, "corrected": score, "gain": overall - score,
                                     "day_block_gain_ci95": uncertainty(frame.date, actual, baseline, estimate, 1),
                                     "seven_day_block_gain_ci95": uncertainty(frame.date, actual, baseline, estimate, 7),
                                     "affected_rows": int(np.count_nonzero(adjustment)), "segments": by_segment}
    return {"scores": scores, "rows": len(frame), "fit_dates_by_bin": fitted_days,
            "primary_rows": int(primary.sum()), "primary_dates": int(frame.loc[primary, "date"].nunique())}


def run(work, team, earlier):
    started = perf_counter()
    holidays, events = protected_days(work)
    baseline, rows = evaluate(work, team, earlier)
    archive = history_and_baselines(work, earlier, rows)
    result = {"minimum_gain": MIN_GAIN, "primary_target_selected_from_prior_abc_diagnostic": True, "folds": {}}
    for fold, (cutoff, _) in WINDOWS.items():
        target = rows.loc[rows.fold.eq(fold)].rename(columns={"learned": "forecast"})
        report_fold = report(target, archive, pd.Timestamp(cutoff), holidays, events, target.station_shift.to_numpy())
        scores = report_fold["scores"]
        if (scores["raw"]["primary_2_to_5"]["baseline"] != baseline["folds"][fold]["uncalibrated"]["learned"]
                or scores["station_calibrated"]["primary_2_to_5"]["baseline"] != baseline["folds"][fold]["prior_only_calibration"]["learned"]):
            raise ValueError("The raw and station-calibrated baselines must replay exactly.")
        result["folds"][fold] = report_fold

    previous = archive.assign(error=archive.y_log - archive.local_stack)
    for cutoff, end in (("2025-07-31", "2025-08-31"), ("2025-08-31", "2025-10-31")):
        origin = pd.Timestamp(cutoff)
        target = archive.loc[archive.origin.eq(origin) & archive.date.le(end)]
        prior_days = previous.loc[previous.origin.lt(origin) & previous.date.le(origin), "date"].nunique()
        shift = historical_station_shift(previous, target, origin)[0] if prior_days >= 60 else None
        result[f"secondary_{cutoff}"] = report(target, archive, origin, holidays, events, shift)
    result["elapsed_seconds"] = round(perf_counter() - started, 2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team", type=Path)
    parser.add_argument("earlier", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    report = run(args.work_dir, args.team, args.earlier)
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (args.work_dir / "sincon_late_low_count_calibration_results.json").write_text(output)
