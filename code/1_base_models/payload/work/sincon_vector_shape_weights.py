"""Compare two fixed weights for ordinary-date vector XGBoost shape corrections."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sincon_archived_intraday_anomalies import CALENDAR, DAY, MAX_SHIFT, WEIGHT, vector_inputs, vector_signal
from sincon_munhak_ordinary import WARM
from sincon_station_state import KEY, WINDOWS
from sincon_tabicl_day_weight import BASE_WEIGHT, exclude_retail, fit_weight, guarded_weight, load_replay, split_check
from sincon_weather_sensitivity import score
from sincon_weekly_variability import protected_days


WEIGHTS = (WEIGHT, .20, .30)
CASES = (*WINDOWS, "summer", *dict(WARM))
LEADS = ((1, 30), (31, 60), (61, 90), (91, 123))
MATURE_EXCLUSIONS = (3107, 3108, 3109)


def read_source(work):
    columns = [*KEY, "fold", "reference", "shape"]
    rows = pd.read_csv(work / "sincon_vector_boost_validation.csv.gz", usecols=columns, parse_dates=["date"])
    if rows.duplicated(["fold", *KEY]).any() or not np.isfinite(rows[["shape", "reference"]]).all().all():
        raise ValueError("Invalid archived vector shape predictions.")
    return {label: block.set_index(KEY).eval("shape - reference") for label, block in rows.groupby("fold")}


def vector_replay(args, frames):
    work = args.work_dir
    inputs = vector_inputs(args)
    for label, frame in frames.items():
        started = perf_counter()
        origin = frame.origin.iloc[0]
        source = work / "sincon_summer_replay_validation.csv.gz" if label == "summer" else args.rolling_archive / f"origin_{label}.csv.gz"
        prior = pd.read_csv(source, usecols=[*KEY, "conditional_stack", "y_log"], parse_dates=["date"]).set_index(KEY)
        keys = pd.MultiIndex.from_frame(frame[KEY])
        truth = prior.y_log.reindex(keys)
        if not np.allclose(truth, frame.y_log, atol=1e-8, rtol=0.):
            raise ValueError(f"The archived conditional precursor must match the official labels for {label}.")
        signal = vector_signal(inputs, origin, label, prior.conditional_stack.reindex(keys))["vector_xgb"]
        print(f"Replayed {label} in {perf_counter() - started:.1f}s.", flush=True)
        yield label, signal.droplevel("fold").rename("raw_delta")


def offsets(frame, source):
    selected = frame.eligible & ~exclude_retail(frame)
    regular = frame.loc[selected, [*KEY, "baseline"]].copy()
    regular["month"] = regular.date.dt.to_period("M")
    regular["weekday"] = regular.date.dt.day_of_week
    raw = source.reindex(pd.MultiIndex.from_frame(frame[KEY])).to_numpy()
    if not np.isfinite(raw).all():
        raise ValueError("A raw vector anomaly is required for every scored key.")
    regular["shape"] = raw[selected]
    regular["shape"] -= regular.groupby(DAY)["shape"].transform("mean")
    anomaly = regular["shape"] - regular.groupby([*CALENDAR, "hour"])["shape"].transform("mean")
    results = {}
    for weight in WEIGHTS:
        shift = weight * anomaly
        legal = np.divide(regular.baseline, -shift, out=np.ones(len(regular)), where=shift.lt(0.)).clip(0., 1.)
        cap = MAX_SHIFT / np.maximum(shift.abs(), MAX_SHIFT)
        scale = pd.Series(np.minimum(legal, cap), index=regular.index)
        scale = scale.groupby([regular[col] for col in CALENDAR]).transform("min")
        change = shift * scale
        daily = change.groupby([regular.date, regular.station_id]).sum().abs().max()
        recurring = change.groupby([regular[col] for col in (*CALENDAR, "hour")]).sum().abs().max()
        candidate = frame.baseline.to_numpy() + change.reindex(frame.index, fill_value=0.).to_numpy()
        if (daily > 1e-12 or recurring > 1e-12 or (candidate < -1e-12).any()
                or change.abs().max() > MAX_SHIFT + 1e-12):
            raise ValueError("The vector candidate must preserve protected rows and both log-mean constraints.")
        results[f"{weight:.0%}"] = candidate
    return results


def evaluate(frame, splits=False):
    if frame.duplicated(KEY).any():
        raise ValueError("Pooled vector validation needs disjoint station-hour keys.")
    actual, incumbent = frame.y_log.to_numpy(), frame["10%"].to_numpy()
    result = {"rows": len(frame), "dates": int(frame.date.nunique()),
              "without_vector": score(actual, incumbent, frame.baseline.to_numpy(), frame.date)["candidate"]}
    for weight in WEIGHTS[1:]:
        name = f"{weight:.0%}"
        comparison = score(actual, incumbent, frame[name].to_numpy(), frame.date)
        result[name] = {**comparison, "gain": comparison["reference"] - comparison["candidate"]}
        if splits:
            result[name]["month_halves"] = split_check(frame.assign(baseline=frame["10%"]), name)
    return result


def run(args):
    replay = load_replay(args, protected_days(args.work_dir))
    expected = json.loads((args.work_dir / "sincon_tabicl_day_weight_results.json").read_text())
    models = read_source(args.work_dir)
    if args.vector_cache and args.vector_cache.exists():
        cached = pd.read_pickle(args.vector_cache)
        for label, block in cached.groupby("label"):
            models[label] = block.set_index(KEY).raw_delta
    frames = {}
    for label in CASES:
        frame = replay.loc[replay.label.eq(label)].reset_index(drop=True)
        origin = frame.origin.iloc[0]
        prior_weight, _ = fit_weight(replay, origin)
        extra = (guarded_weight(frame.date, origin, prior_weight) / BASE_WEIGHT - 1.) * frame.log_delta
        frame["baseline"] = (frame.baseline + extra).clip(lower=0.)
        actual = np.sqrt(np.square(frame.y_log - frame.baseline).mean())
        if not np.isclose(actual, expected[label]["guarded"]["candidate"], atol=1e-10, rtol=0.):
            raise ValueError(f"Guarded TabICL incumbent did not reproduce for {label}.")
        frames[label] = frame
    missing = {label: frame for label, frame in frames.items() if label not in models}
    if missing and args.external:
        models.update(vector_replay(args, missing))
        if args.vector_cache:
            pd.concat([models[label].reset_index().assign(label=label) for label in CASES if label not in WINDOWS],
                      ignore_index=True).to_pickle(args.vector_cache)

    prior_checks = {}
    for filename in ("sincon_archived_intraday_anomalies_results.json", "sincon_archived_intraday_anomalies_warm_results.json"):
        prior_checks.update(json.loads((args.work_dir / filename).read_text())["cases"])
    report, cases = {}, {}
    for label, frame in frames.items():
        if label not in models:
            continue
        cases[label] = frame.assign(**offsets(frame, models[label]))
        report[label] = evaluate(cases[label])
    pools = {"disjoint_A_C": pd.concat([cases["A"], cases["C"]], ignore_index=True)}
    if "summer" in cases:
        pools["strict_summer_mature_68"] = cases["summer"].loc[~cases["summer"].station_id.isin(MATURE_EXCLUSIONS)]
    if all(label in cases for label, _ in WARM):
        pools["disjoint_warm"] = pd.concat([cases[label].loc[cases[label].date.dt.month.eq(month)] for label, month in WARM],
                                           ignore_index=True)
    for label, frame in pools.items():
        report[label] = evaluate(frame, splits=True)
        leads = (frame.date - frame.origin).dt.days
        report[label]["lead_days"] = {f"{start}_{end}": evaluate(frame.loc[leads.between(start, end)])
                                      for start, end in LEADS if leads.between(start, end).any()}
    for label, actual in report.items():
        check = "strict_summer" if label == "summer" else label
        known = prior_checks[check]["all"]["vector_xgb"]["candidate"]
        if not np.isclose(actual["20%"]["reference"], known, atol=1e-10, rtol=0.):
            raise ValueError(f"Original 10% vector candidate did not reproduce for {label}.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "archive", "tabicl_oof", "retail_oof", "summer_tabicl", "results"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--external", type=Path)
    parser.add_argument("--rolling-archive", type=Path)
    parser.add_argument("--vector-cache", type=Path)
    args = parser.parse_args()
    if bool(args.external) != bool(args.rolling_archive):
        parser.error("A vector model replay needs the external features and archived rolling conditional forecasts.")
    report = run(args)
    args.results.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for label, values in report.items():
        print(label, {name: values[name]["gain"] for name in ("20%", "30%")}, flush=True)


if __name__ == "__main__":
    main()
