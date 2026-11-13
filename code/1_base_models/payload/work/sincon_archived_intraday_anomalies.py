"""Replay ordinary-date intraday anomalies from independent hourly model archives."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_munhak_ordinary import WARM
from sincon_station_state import KEY, WINDOWS
from sincon_tabicl_day_weight import BASE_WEIGHT, exclude_retail, fit_weight, guarded_weight, load_replay
from sincon_weather_sensitivity import score
from sincon_weekly_variability import protected_days


WEIGHT = .10
MAX_SHIFT = .06
DAY = KEY[:2]
CALENDAR = ["station_id", "month", "weekday"]
SOURCES = {
    "poisson": ("count_distribution", "reference", "poisson_mean_log", 1.),
    "tweedie": ("count_distribution", "reference", "tweedie_mean_log", 1.),
    "dynamic_bands": ("dynamic_levels", "reference", "bands_joint_state", 2.),
    "vector_xgb": ("vector_boost", "reference", "shape", 1.),
    "smoothed_ridge": ("smooth_hours", "reference", "smooth_only", 4.),
    "gru": ("recurrent", "joint", "gru", 1.),
    "tabm": ("tabm", "joint", "tabm", 1.),
}


def archive_design(work):
    forecasts = {}
    columns = [*KEY, "fold"]
    for source, (file, reference, candidate, scale) in SOURCES.items():
        rows = pd.read_csv(work / f"sincon_{file}_validation.csv.gz", usecols=[*columns, reference, candidate], parse_dates=["date"])
        if rows.duplicated(columns).any():
            raise ValueError(f"Repeated chronological validation keys in {source}.")
        forecasts[source] = ((rows[candidate] - rows[reference]) * scale).set_axis(pd.MultiIndex.from_frame(rows[columns]))
    return forecasts


def proposal(frame, models):
    active = frame.eligible & ~exclude_retail(frame)
    regular = frame.loc[active, [*KEY, "baseline"]].copy()
    regular["month"] = regular.date.dt.to_period("M")
    regular["weekday"] = regular.date.dt.dayofweek
    keys = pd.MultiIndex.from_frame(frame[KEY].assign(fold=frame.label))
    result = {}
    for source, deltas in models.items():
        aligned = deltas.reindex(keys).to_numpy()
        if not np.isfinite(aligned).all():
            raise ValueError(f"Archived forecast coverage differs in {source}.")
        regular["raw"] = aligned[active]
        centered = regular.raw - regular.groupby(DAY).raw.transform("mean")
        regular["centered"] = centered
        shift = WEIGHT * (centered - regular.groupby([*CALENDAR, "hour"]).centered.transform("mean"))
        # Shared scales keep both station-day and recurring station-hour means fixed.
        legal = np.divide(regular.baseline, -shift, out=np.ones(len(regular)), where=shift.lt(0.)).clip(0., 1.)
        cap = MAX_SHIFT / np.maximum(shift.abs(), MAX_SHIFT)
        regular["scale"] = np.minimum(legal, cap)
        offset = shift * regular.groupby(CALENDAR).scale.transform("min")
        daily = offset.groupby([regular.date, regular.station_id]).sum()
        profile = offset.groupby([regular.station_id, regular.month, regular.weekday, regular.hour]).sum()
        if daily.abs().max() > 1e-12 or profile.abs().max() > 1e-12:
            raise ValueError("A valid anomaly must preserve both daily and calendar-hour levels.")
        result[source] = offset.reindex(frame.index, fill_value=0.).to_numpy()
    return result


def evaluate(frame, sources=SOURCES):
    if frame.duplicated(KEY).any():
        raise ValueError("Pooled validation must use nonoverlapping station-hour keys.")
    masks = {"all": np.ones(len(frame), dtype=bool), "ordinary_late": frame.eligible & ~exclude_retail(frame) & frame.hour.ge(21)}
    result = {}
    for scope, mask in masks.items():
        rows = frame.loc[mask]
        actual, base, dates = rows.y_log.to_numpy(), rows.baseline.to_numpy(), rows.date.to_numpy()
        result[scope] = {}
        for source in sources:
            delta = rows[source].to_numpy()
            scores = score(actual, base, base + delta, dates)
            result[scope][source] = {**scores, "gain": scores["reference"] - scores["candidate"],
                                      "changed_rows": int(np.count_nonzero(delta))}
    return result


def vector_inputs(args):
    from sincon_joint_candidate import setup

    work = args.work_dir
    data, _, daily, features, stations, _, _ = setup(args.team, args.external, work / "sincon_kbo_game_data.csv",
                                                      work / "mobility_monthly", work / "sincon_station_addresses.csv", 6)
    station_ids = {station: index for index, station in enumerate(stations.index)}
    daily = daily + [name for name in features if name.startswith("daily_arrivals_")]
    hourly = [name for name in features if name.startswith(("arrivals_", "hourly_share_"))]
    return data, daily, station_ids, hourly


def vector_signal(inputs, origin, label, reference):
    from sincon_vector_boost import forecast

    data, daily, station_ids, hourly = inputs
    end = reference.index.get_level_values("date").max()
    history = data.loc[data.date.le(origin)]
    future = data.loc[data.date.gt(origin) & data.date.le(end)]
    target = pd.MultiIndex.from_frame(future[KEY])
    if not target.is_unique or not target.sort_values().equals(reference.index.sort_values()):
        raise ValueError(f"Vector prediction must cover the complete forecast grid in {label}.")
    values = forecast(history, future.assign(boardings=np.nan), daily, station_ids, hourly, 6, 650, "shape")
    profile = reference.reindex(target).to_numpy().reshape(-1, 18)
    change = values - values.mean(axis=1, keepdims=True) - (profile - profile.mean(axis=1, keepdims=True))
    keyed = pd.MultiIndex.from_frame(future[KEY].assign(fold=label))
    return {"vector_xgb": pd.Series(change.ravel(), index=keyed)}


def vector_replays(args, archive, expected):
    work = args.work_dir
    inputs = vector_inputs(args)
    results, warm = {}, []
    for label in args.vector_origins:
        frame = archive.loc[archive.label.eq(label)].reset_index(drop=True)
        origin = frame.origin.iloc[0]
        weight, _ = fit_weight(archive, origin)
        extra = (guarded_weight(frame.date, origin, weight) / BASE_WEIGHT - 1.) * frame.log_delta
        frame["baseline"] = (frame.baseline + extra).clip(lower=0.)
        baseline_score = np.sqrt(np.square(frame.y_log - frame.baseline).mean())
        if not np.isclose(baseline_score, expected[label]["guarded"]["candidate"], atol=1e-10, rtol=0.):
            raise ValueError(f"The guarded baseline does not reproduce in {label}.")
        source = work / "sincon_summer_replay_validation.csv.gz" if label == "summer" else args.rolling_archive / f"origin_{label}.csv.gz"
        reference = pd.read_csv(source, usecols=[*KEY, "conditional_stack", "y_log"], parse_dates=["date"]).set_index(KEY)
        keys = pd.MultiIndex.from_frame(frame[KEY])
        if not np.allclose(reference.y_log.reindex(keys), frame.y_log, atol=1e-8, rtol=0.):
            raise ValueError(f"The conditional precursor differs from the archived labels in {label}.")
        signal = vector_signal(inputs, origin, label, reference.conditional_stack.reindex(keys))
        frame = frame.assign(**proposal(frame, signal))
        case = "strict_summer" if label == "summer" else label
        results[case] = evaluate(frame, signal)
        print(case, results[case]["all"]["vector_xgb"], flush=True)
        if label == "summer":
            mature = frame.loc[~frame.station_id.isin((3107, 3108, 3109))]
            results["strict_summer_mature_68"] = evaluate(mature, signal)
        else:
            month = dict(WARM)[label]
            warm.append(frame.loc[frame.date.dt.month.eq(month)])
    if len(warm) == len(WARM):
        results["disjoint_warm"] = evaluate(pd.concat(warm, ignore_index=True), ("vector_xgb",))
        expected_warm = expected["disjoint_warm"]["guarded"]["candidate"]
        if not np.isclose(results["disjoint_warm"]["all"]["vector_xgb"]["reference"], expected_warm, atol=1e-10, rtol=0.):
            raise ValueError("The disjoint warm baseline must match the guarded replay.")
    return results


def run(args):
    archive = load_replay(args, protected_days(args.work_dir))
    expected = json.loads((args.work_dir / "sincon_tabicl_day_weight_results.json").read_text())
    models = archive_design(args.work_dir)
    cases, results = {}, {}
    for label, frame in archive.loc[archive.label.isin(WINDOWS)].groupby("label", sort=False):
        origin = frame.origin.iloc[0]
        issued, end = map(pd.Timestamp, WINDOWS[label])
        if not frame.origin.eq(issued).all() or frame.date.min() <= origin or frame.date.max() != end:
            raise ValueError(f"Historical forecast origin or end changed in {label}.")
        weight, _ = fit_weight(archive, origin)
        extra = (guarded_weight(frame.date, origin, weight) / BASE_WEIGHT - 1.) * frame.log_delta
        base = (frame.baseline + extra).clip(lower=0.)
        if not np.isclose(np.sqrt(np.square(frame.y_log - base).mean()), expected[label]["guarded"]["candidate"], atol=1e-10, rtol=0.):
            raise ValueError(f"The guarded incumbent must replay exactly in {label}.")
        frame = frame.assign(baseline=base)
        cases[label] = frame.assign(**proposal(frame, models))
        results[label] = evaluate(cases[label])
    results["disjoint_A_C"] = evaluate(pd.concat([cases["A"], cases["C"]], ignore_index=True))
    if args.external:
        results.update(vector_replays(args, archive, expected))
    return {"frozen_weight": WEIGHT, "max_absolute_log_change": MAX_SHIFT, "cases": results,
            "additional_vector_origins": args.vector_origins if args.external else []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "archive", "tabicl_oof", "retail_oof", "summer_tabicl", "results"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--external", type=Path, help="Weather archive for the summer vector shape replays.")
    parser.add_argument("--rolling-archive", type=Path)
    parser.add_argument("--vector-origins", nargs="+", choices=("summer", *dict(WARM)), default=("summer",))
    args = parser.parse_args()
    if args.external and any(label != "summer" for label in args.vector_origins) and not args.rolling_archive:
        parser.error("Rolling vector replays need the matching conditional forecast archive.")
    result = run(args)
    args.results.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for case, metrics in result["cases"].items():
        print(case, {source: round(scores["gain"], 7) for source, scores in metrics["all"].items()}, flush=True)


if __name__ == "__main__":
    main()
