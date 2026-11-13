"""Backtest a fixed, complementary blend on archived SINCON forecasts."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sincon_stadium_concert_analogs import HOURS, STADIUMS, analog_forecast, ordinary_profile, read_inputs, usual_days
from sincon_station_state import KEY, WINDOWS
from sincon_weekly_variability import correction, past_episodes, protected_days, weekly_signal


JOIN = ["fold", *KEY]
TARGET = .150
REQUIRED_GAIN = .003


def load_prior_blend(work):
    columns = [*JOIN, "y_log", "reference", "independent_sources", "with_regime"]
    archive = pd.read_csv(work / "sincon_local_blend_validation.csv.gz", usecols=columns, parse_dates=["date"])
    if archive.duplicated(JOIN).any() or set(archive.fold) != set(WINDOWS):
        raise ValueError("The earlier blend must cover the same unique A/B/C prediction rows.")
    return archive.set_index(JOIN)


def peak_change(forecast, inputs, cutoff):
    train, events, matches, calendar, baseball = inputs
    adjustments = np.zeros(len(forecast), dtype=float)
    target_dates = forecast.date.unique()
    for station in STADIUMS:
        history = train.loc[station].loc[:cutoff]
        ordinary = usual_days(station, history, events, baseball)
        known = matches.loc[matches.station_id.eq(station)]
        prior = known.loc[known.date.le(cutoff) & known.date.isin(history.index)]
        future = known.loc[known.date.gt(cutoff) & known.date.isin(target_dates)]
        for target in future.itertuples(index=False):
            peers = prior.loc[prior.performance.ne(target.performance)]
            rows = forecast.station_id.eq(station) & forecast.date.eq(target.date) & forecast.hour.ge(19)
            if peers.empty or not rows.any():
                continue
            positions = np.searchsorted(HOURS, forecast.loc[rows, "hour"])
            background = ordinary_profile(ordinary, target.date, calendar)[positions]
            baseline = forecast.loc[rows, "base"].to_numpy()
            peak = np.argmax(np.expm1(baseline) - background)
            _, historical_peak = analog_forecast(target, ordinary, history, peers, calendar, timing=True)
            row = np.flatnonzero(rows)[peak]
            adjustments[row] = .5 * max(0., np.log1p(background[peak] + historical_peak) - baseline[peak])
    return adjustments


def mean_neutral(frame, delta):
    center = pd.Series(delta, index=frame.index).groupby(frame.station_id).transform("mean")
    return delta - center.to_numpy()


def historical_station_shift(*args, **kwargs):
    raise RuntimeError("Obsolete Public-scaled historical calibration is blocked")


def rmsle(observed, estimated):
    return float(np.sqrt(np.mean(np.square(observed - np.maximum(estimated, 0)))))


def interval(frame, observed, baseline, estimate):
    dates = (frame.date - frame.date.min()).dt.days // 7
    errors = pd.DataFrame({
        "baseline": np.square(observed - np.maximum(baseline, 0)),
        "candidate": np.square(observed - np.maximum(estimate, 0)),
        "rows": 1,
    }).groupby(dates).sum().to_numpy()
    rng = np.random.default_rng(0)
    totals = errors[rng.integers(len(errors), size=(2000, len(errors)))].sum(axis=1)
    gains = np.sqrt(totals[:, 0] / totals[:, 2]) - np.sqrt(totals[:, 1] / totals[:, 2])
    return np.quantile(gains, [.025, .975]).tolist()


def evaluate(work, team, earlier):
    holidays, events = protected_days(work)
    current, prior = past_episodes(work / "sincon_local_validation.csv.gz", earlier, holidays, events)
    blend = load_prior_blend(work)
    concert = read_inputs(work, team)
    raw = pd.read_csv(team / "data/train.csv", usecols=[*KEY, "boardings"], parse_dates=["date"]).set_index(KEY).boardings
    results, saved = {}, []
    for fold, (origin, end) in WINDOWS.items():
        started = perf_counter()
        cutoff = pd.Timestamp(origin)
        frame = weekly_signal(current[fold], holidays, events).merge(current[fold], on=KEY, validate="one_to_one")
        dates = pd.date_range(cutoff + pd.Timedelta(days=1), end)
        if len(frame) != len(dates) * 71 * 18 or not frame.date.drop_duplicates().reset_index(drop=True).equals(pd.Series(dates)):
            raise ValueError(f"Unexpected forecast grid in {fold}.")
        keys = pd.MultiIndex.from_frame(frame[KEY])
        observed = frame.y_log.to_numpy()
        baseline = frame.base.to_numpy()
        components = blend.loc[fold].reindex(keys)
        if not (
            np.allclose(components.reference, baseline, atol=1e-8, rtol=0)
            and np.allclose(components.y_log, observed, atol=1e-8, rtol=0)
            and np.allclose(np.log1p(raw.reindex(keys)), observed, atol=1e-8, rtol=0)
        ):
            raise ValueError(f"Component forecasts or labels disagree in {fold}.")
        weekly, fitted_days = correction(prior, frame, cutoff)
        peak = peak_change(frame, concert, cutoff)
        if (peak[frame.eligible] != 0.).any():
            raise ValueError("The routine weekly and special concert corrections must use disjoint rows.")
        concert_days = pd.MultiIndex.from_frame(concert[2][KEY[:2]])
        ordinary_place = ~pd.MultiIndex.from_frame(frame[KEY[:2]]).isin(concert_days)
        ordinary = np.where(ordinary_place, components.with_regime.to_numpy() - baseline, 0.)
        residual = ordinary + weekly.to_numpy() + peak
        estimate = baseline + residual
        fixed_levels = baseline + mean_neutral(frame, residual)
        # Include the same fixed additions on the older no-break combination
        # for sensitivity, rather than choosing a different blend per fold.
        without_break = baseline + np.where(ordinary_place, components.independent_sources.to_numpy() - baseline, 0.)
        without_break += weekly.to_numpy() + peak
        scores = {
            "local_stack": rmsle(observed, baseline),
            "previous_regime_blend": rmsle(observed, components.with_regime.to_numpy()),
            "fixed_blend": rmsle(observed, estimate),
            "fixed_blend_with_station_levels_preserved": rmsle(observed, fixed_levels),
            "fixed_blend_without_break": rmsle(observed, without_break),
        }
        historical_shift, historical_days = historical_station_shift(prior, frame, cutoff)
        calibrated = {
            "baseline": rmsle(observed, baseline + historical_shift),
            "candidate": rmsle(observed, estimate + historical_shift),
            "station_levels_preserved": rmsle(observed, fixed_levels + historical_shift),
            "days": historical_days,
        }
        results[fold] = {
            "rmsle": scores,
            "chronological_calibration_proxy": calibrated,
            "weekly_calibration_days": fitted_days,
            "concert_rows": int(np.count_nonzero(peak)),
            "rows": len(frame),
            "paired_week_gain_interval": interval(frame, observed, baseline, estimate),
            "seconds": round(perf_counter() - started, 2),
        }
        saved.append(frame[KEY].assign(
            fold=fold, y_log=observed, local_stack=baseline, fixed_blend=estimate,
            fixed_blend_with_station_levels_preserved=fixed_levels,
            fixed_blend_without_break=without_break,
        ))
        print(json.dumps({fold: results[fold]}, allow_nan=False), flush=True)
    rows = pd.concat(saved, ignore_index=True)
    ac = rows.loc[rows.fold.isin(["A", "C"])]
    a = set(ac.loc[ac.fold.eq("A"), "date"])
    c = set(ac.loc[ac.fold.eq("C"), "date"])
    if a & c:
        raise ValueError("A and C are expected to be disjoint.")
    names = ("local_stack", "fixed_blend", "fixed_blend_with_station_levels_preserved", "fixed_blend_without_break")
    disjoint = {name: rmsle(ac.y_log.to_numpy(), ac[name].to_numpy()) for name in names}
    return {
        "folds": results,
        "disjoint_ac": disjoint,
        "local_target": TARGET,
        "minimum_adoption_gain": REQUIRED_GAIN,
        "target_folds": [fold for fold, report in results.items() if report["rmsle"]["fixed_blend"] <= TARGET],
        "adoption_folds": [
            fold for fold, report in results.items()
            if report["rmsle"]["local_stack"] - report["rmsle"]["fixed_blend"] >= REQUIRED_GAIN
        ],
    }, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("earlier", type=Path)
    parser.add_argument("team_dir", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    report, rows = evaluate(args.work_dir, args.team_dir, args.earlier)
    (args.work_dir / "sincon_complementary_blend_results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    if args.predictions:
        rows.to_csv(args.predictions, index=False, float_format="%.9f")
    print(json.dumps({k: v for k, v in report.items() if k != "folds"}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
