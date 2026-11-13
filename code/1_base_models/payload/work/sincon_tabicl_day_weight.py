"""Validate stronger TabICL daily deviations with strictly earlier forecast errors."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_munhak_ordinary import WARM
from sincon_network_level_audit import STABLE_EXCLUSIONS, ordinary
from sincon_ordinary_shape_calibration import latest_validation
from sincon_songdo_schedule import save_submission
from sincon_spectator_month_split import gain_summary, month_halves
from sincon_station_state import KEY
from sincon_tabicl_submission import read_delta
from sincon_weather_sensitivity import score
from sincon_weekly_variability import protected_days


ORIGIN = pd.Timestamp("2026-06-30")
FEATURE_KEYS = ["label", "origin", *KEY]
BASE_WEIGHT = .10
FALLBACK = .20
MAX_WEIGHT = .30
METHODS = ("fixed_20", "fixed_30", "chronological", "guarded")


def exclude_retail(rows):
    return rows.station_id.eq(3131) & rows.date.dt.day_of_week.eq(6)


def verify_signal(rows, values):
    if not np.isfinite(values).all() or np.any(values[exclude_retail(rows)]):
        raise ValueError("The proposed additional TabICL signal must be finite and preserve Dongchun Sundays.")
    day = pd.DataFrame({"date": rows.date.to_numpy(), "station_id": rows.station_id.to_numpy(), "signal": values})
    by_day = day.groupby(KEY[:2]).signal.agg(["min", "max", "mean"])
    month = by_day.index.get_level_values("date").to_period("M")
    weekday = by_day.index.get_level_values("date").day_of_week
    if ((by_day["max"] - by_day["min"]).abs().max() > 1e-12
            or by_day["mean"].groupby([by_day.index.get_level_values("station_id"), month, weekday]).sum().abs().max() > 1e-11):
        raise ValueError("Scaling the TabICL signal must preserve hourly shape and station-month-weekday level.")


def fit_weight(archive, origin):
    prior = archive.loc[archive.date.le(origin) & archive.origin.lt(origin) & archive.log_delta.ne(0.)]
    prior = prior.sort_values(["reliability", "origin"]).drop_duplicates(KEY, keep="last")
    if not prior.date.gt(prior.origin).all():
        raise ValueError("Weights must be trained with realizations from a prior-issued forecast.")
    days = int(prior.date.nunique())
    if days < 24:
        return FALLBACK, {"prior_dates": days, "fallback": True}
    time_weight = np.exp2(-(origin - prior.date).dt.days.to_numpy() / 180.) * prior.reliability.to_numpy()
    delta = prior.log_delta.to_numpy()
    residual = (prior.y_log - prior.baseline).clip(-.60, .60).to_numpy()
    energy = float(np.dot(time_weight * delta, delta))
    alignment = float(np.dot(time_weight * delta, residual))
    # Use twenty-four equivalent dates at the incumbent weight as shrinkage.
    extra = alignment / (energy * (1. + 24. / days))
    weight = float(np.clip(BASE_WEIGHT * (1. + extra), BASE_WEIGHT, MAX_WEIGHT))
    return weight, {"prior_dates": days, "fallback": False, "latest_prior_date": str(prior.date.max().date()),
                    "unbounded_weight_after_shrinkage": float(BASE_WEIGHT * (1. + extra))}


def guarded_weight(dates, origin, weight):
    months_ahead = 12 * (dates.dt.year - origin.year) + dates.dt.month - origin.month
    factor = np.select([months_ahead.le(2), months_ahead.eq(3)], [1., .5], default=0.)
    return BASE_WEIGHT + factor * (weight - BASE_WEIGHT)


def evaluate(frame):
    base, actual = frame.baseline.to_numpy(), frame.y_log.to_numpy()
    delta = frame.log_delta.to_numpy()
    moment = float(np.dot(delta, delta))
    report = {method: score(actual, base, frame[method].to_numpy(), frame.date) for method in METHODS}
    report["heldout_best_weight_diagnostic"] = float(BASE_WEIGHT * (1. + np.dot(delta, actual - base) / moment))
    return report


def split_check(frame, method):
    actual, base, proposed = (frame[name].to_numpy() for name in ("y_log", "baseline", method))
    daily = frame[["date"]].assign(base=(actual - base)**2, proposed=(actual - proposed)**2, n=1).groupby("date").sum()
    assignment = month_halves(daily.index, 10_000, 116).astype(float)
    sums, n = daily[["base", "proposed"]].to_numpy(), daily.n.to_numpy()
    first_sums, first_n = assignment @ sums, assignment @ n
    first = np.sqrt(first_sums / first_n[:, None])
    second = np.sqrt((sums.sum(axis=0) - first_sums) / (n.sum() - first_n)[:, None])
    gains = np.column_stack((first[:, 0] - first[:, 1], second[:, 0] - second[:, 1]))
    return {**gain_summary(gains.ravel()), "both_improve_share": float(np.all(gains > 1e-12, axis=1).mean())}


def load_replay(args, protected):
    prior = pd.read_pickle(args.tabicl_oof)
    strict = pd.read_pickle(args.summer_tabicl)
    if len(strict) != 157_194 or not strict.label.eq("summer").all() or not strict.origin.eq("2025-06-30").all():
        raise ValueError("Strict TabICL must be the complete 123-day forecast issued on 2025-06-30.")
    deltas = pd.concat([prior, strict], ignore_index=True)
    raw = pd.read_csv(args.team / "data/train.csv", parse_dates=["date"], usecols=[*KEY, "boardings"])
    history = raw.assign(observed=np.log1p(raw.pop("boardings")))
    replay = latest_validation(args, history, protected, tabicl=deltas)
    replay = replay.merge(deltas, on=FEATURE_KEYS, how="inner", validate="one_to_one")
    if len(replay) != len(deltas):
        raise ValueError("Every adjustment must be paired with the exact current archived forecast.")
    replay.loc[exclude_retail(replay), "log_delta"] = 0.
    for _, rows in replay.groupby("label"):
        verify_signal(rows, rows.log_delta.to_numpy())
    return replay


def final_candidate(args, grid, incumbent, weight, protected):
    tabicl_manifest = json.loads(args.tabicl_report.read_text())
    prior_manifest = json.loads(args.incumbent_report.read_text())
    if hashlib.sha256(args.baseline.read_bytes()).hexdigest() != prior_manifest["export"]["sha256"]:
        raise ValueError("The current forecast must match the verified Geomdan morning candidate.")
    delta = read_delta(args.final_delta, grid, tabicl_manifest["sha256"]["delta"], full=True)
    delta = np.where(exclude_retail(grid), 0., delta)
    verify_signal(grid, delta)
    eligible = ordinary(grid, protected) & ~grid.station_id.isin(STABLE_EXCLUSIONS)
    weight_by_row = guarded_weight(grid.date, ORIGIN, weight)
    additional = delta * (weight_by_row / BASE_WEIGHT - 1.)
    verify_signal(grid, additional)
    changed = additional != 0.
    if np.any(changed & ~eligible) or not changed.any() or weight <= BASE_WEIGHT:
        raise ValueError("Only supported ordinary station-days may receive the increased weight.")
    values = incumbent.boardings.to_numpy(copy=True)
    values[changed] = np.expm1(np.log1p(values[changed]) + additional[changed])
    save_submission(args.submission, incumbent, grid, values)
    months = grid.date.dt.strftime("%Y-%m")
    schedule = grid.assign(weight=weight_by_row).groupby(months).weight.first().to_dict()
    return {"submission": str(args.submission.resolve()), "weight_by_month": schedule, "rows": len(grid),
            "changed_rows": int(changed.sum()), "changed_station_days": len(grid.loc[changed, KEY[:2]].drop_duplicates()),
            "max_additional_log_adjustment": float(np.max(np.abs(additional))),
            "sha256": hashlib.sha256(args.submission.read_bytes()).hexdigest()}


def run(args):
    protected = protected_days(args.work_dir)
    replay = load_replay(args, protected)
    prior_metrics = json.loads((args.work_dir / "sincon_geomdan_morning_anchor_results.json").read_text())
    strict_metrics = json.loads((args.work_dir / "sincon_strict_season_final_audit_results.json").read_text())
    cases, report = {}, {}
    for label, frame in replay.groupby("label", sort=False):
        origin = frame.origin.iloc[0]
        actual_rmsle = np.sqrt(np.square(frame.y_log - frame.baseline).mean())
        exact = (strict_metrics["all_stations"]["full_rmsle"]["chuseok"] if label == "summer"
                 else prior_metrics[label]["all"]["relative_peak_share"]["candidate"])
        if not np.isclose(actual_rmsle, exact, atol=1e-10, rtol=0.):
            raise ValueError(f"The incumbent RMSLE did not reproduce for {label}.")
        weight, support = fit_weight(replay, origin)
        amplitudes = (.20, .30, weight, guarded_weight(frame.date, origin, weight))
        frame = frame.assign(**{name: (frame.baseline + (amp / BASE_WEIGHT - 1.) * frame.log_delta).clip(lower=0.)
                                for name, amp in zip(METHODS, amplitudes)})
        cases[label] = frame
        report[label] = {"origin": str(origin.date()), "chronological_weight": weight, **support, **evaluate(frame)}
    warm = pd.concat([cases[label].loc[cases[label].date.dt.month.eq(month)] for label, month in WARM])
    groups = {"disjoint_warm": warm, "disjoint_A_C": pd.concat([cases["A"], cases["C"]]),
              "strict_established_68": cases["summer"].loc[~cases["summer"].station_id.isin((3107, 3108, 3109))]}
    for label, frame in groups.items():
        if frame.duplicated(KEY).any():
            raise ValueError("An aggregate validation group must have distinct station-hour keys.")
        report[label] = {**evaluate(frame), "month_halves": {method: split_check(frame, method)
                                                           for method in ("chronological", "guarded")}}
    report["strict_established_68"]["by_month"] = {
        str(month): evaluate(frame) for month, frame in groups["strict_established_68"].groupby(groups["strict_established_68"].date.dt.month)
    }
    weight, support = fit_weight(replay, ORIGIN)
    report["final_fit"] = {"origin": str(ORIGIN.date()), "chronological_weight": weight, **support}
    if args.submission:
        adopted = [report[case]["guarded"] for case in ("A", "B", "C", *groups)]
        if any(item["candidate"] >= item["reference"] for item in adopted):
            raise ValueError("The chronological weight must improve the nonoverlapping and same-season checks.")
        grid = pd.read_csv(args.team / "data/test.csv", parse_dates=["date"], usecols=["example_id", *KEY])
        current = pd.read_csv(args.baseline, float_precision="round_trip")
        if len(grid) != 157_194 or not grid.example_id.equals(current.example_id) or not grid.example_id.is_unique:
            raise ValueError("Use all official test IDs in original order.")
        report["export"] = final_candidate(args, grid, current, weight, protected)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "archive", "tabicl_oof", "retail_oof", "summer_tabicl", "results"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    for name in ("baseline", "incumbent-report", "final-delta", "tabicl-report", "submission"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    export = (args.baseline, args.incumbent_report, args.final_delta, args.tabicl_report, args.submission)
    if any(export) and not all(export):
        parser.error("Export requires the checked incumbent and TabICL artifacts, and a new output path.")
    report = run(args)
    args.results.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for label in ("A", "B", "C", "disjoint_A_C", "disjoint_warm", "strict_established_68", "final_fit", "export"):
        if label in report:
            print(label, report[label].get("guarded", report[label]), flush=True)


if __name__ == "__main__":
    main()
