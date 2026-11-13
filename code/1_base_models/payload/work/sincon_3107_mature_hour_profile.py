"""Test mature daytime boarding profiles for Geomdan Lake Park (3107)."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sincon_complementary_blend import interval, rmsle
from sincon_geomdan_opening_control import OPENED
from sincon_overlay_candidate import replay, trained_inputs
from sincon_station_state import KEY
from sincon_xgb_conditioned import load_episodes


STATION = 3107
NEIGHBORS = [3108, 3109]
HOURS = np.arange(6, 21)
METHODS = ("opening_profile", "recent_profile", "recent_relative_to_neighbors")
MINIMUM_SAMPLE = 4


def profiles(rows, column):
    local = rows.loc[rows.station_id.isin([STATION, *NEIGHBORS]) & rows.hour.isin(HOURS)]
    cube = local.pivot(index="date", columns=["station_id", "hour"], values=column)
    cube = cube.reindex(columns=pd.MultiIndex.from_product([[STATION, *NEIGHBORS], HOURS]))
    array = cube.to_numpy().reshape(len(cube), 3, len(HOURS))
    if not np.isfinite(array).all():
        raise ValueError("The three Geomdan stations require complete daytime observations once open.")
    return cube.index, array, array - array.mean(axis=2, keepdims=True)


def ordinary(dates, protected):
    holidays, events = protected
    blocked = dates.isin(holidays)
    for station in (STATION, *NEIGHBORS):
        days = pd.MultiIndex.from_arrays([dates, np.full(len(dates), station)], names=KEY[:2])
        blocked |= days.isin(events)
    return ~blocked


def weekday_group(dates):
    # Monday through Thursday share commuter patterns. Friday, Saturday and
    # Sunday retain separate profiles when prior support permits.
    return np.maximum(dates.dayofweek.to_numpy() - 3, 0)


def template(reference, source, matches, future_groups, historical_groups):
    change = np.zeros_like(reference)
    groups = []
    for group in np.unique(future_groups):
        target = future_groups == group
        prior = matches & (historical_groups == group)
        if prior.sum() < MINIMUM_SAMPLE:
            prior = matches & ((historical_groups >= 2) == (group >= 2))
        if prior.sum() < MINIMUM_SAMPLE:
            continue
        shift = np.median(source[prior], axis=0) - np.median(reference[target], axis=0)
        centered = .5 * (shift - shift.mean())
        change[target] = centered * min(1., .25 / max(np.abs(centered).max(), 1e-12))
        groups.append(int(group))
    return change, groups


def adjustment(history, future, origin, protected):
    if history.date.max() > origin or future.date.min() <= origin or {"boardings", "y_log"} & set(future):
        raise ValueError("Fit the profile on prior observations and unlabeled future inputs.")
    past_dates, _, observed = profiles(history, "observed")
    future_dates, predicted, model = profiles(future, "combined")
    historical_ordinary = ordinary(past_dates, protected)
    target = ordinary(future_dates, protected)
    after_ramp = past_dates >= OPENED + pd.Timedelta(days=14)
    recent = historical_ordinary & after_ramp & (past_dates > origin - pd.Timedelta(days=56))
    opening = historical_ordinary & ~after_ramp
    supports = {"recent_days": int(recent.sum()), "opening_days": int(opening.sum()), "weekday_groups": {}}
    model_relative = model[:, 0] - model[:, 1:].mean(axis=1)
    observed_relative = observed[:, 0] - observed[:, 1:].mean(axis=1)
    specs = ((model[:, 0], observed[:, 0], opening), (model[:, 0], observed[:, 0], recent),
             (model_relative, observed_relative, recent))
    key = pd.MultiIndex.from_frame(future[["date", "hour"]])
    scope = future.station_id.eq(STATION).to_numpy() & future.hour.isin(HOURS).to_numpy()
    dates_and_hours = pd.MultiIndex.from_product([future_dates, HOURS], names=["date", "hour"])
    deltas = {}
    for method, (reference, source, matches) in zip(METHODS, specs):
        daily, supports["weekday_groups"][method] = template(reference[target], source, matches,
                                                             weekday_group(future_dates[target]), weekday_group(past_dates))
        change = np.zeros_like(predicted[:, 0])
        base = np.maximum(predicted[target, 0], 0.)
        legal = np.divide(base, -daily, out=np.ones_like(daily), where=daily < -1e-12)
        change[target] = daily * np.minimum(legal.min(axis=1), 1.)[:, None]
        if np.abs(change.sum(axis=1)).max() > 1e-10:
            raise ValueError("Daytime shape changes must preserve every station-day log level.")
        delta = np.zeros(len(future))
        delta[scope] = pd.Series(change.ravel(), index=dates_and_hours).reindex(key[scope]).to_numpy()
        deltas[method] = delta
    return deltas, supports


def score(frame):
    actual = frame.y_log.to_numpy()
    base = frame.combined.clip(lower=0.).to_numpy()
    local = frame.station_id.eq(STATION).to_numpy()
    early = local & frame.hour.isin(HOURS).to_numpy()
    baseline = rmsle(actual, base)
    result = {"rows": len(frame), "days": int(frame.date.nunique()), "baseline": baseline,
              "3107_baseline": rmsle(actual[local], base[local]), "3107_daytime_baseline": rmsle(actual[early], base[early])}
    for method in METHODS:
        delta = frame[method].to_numpy()
        estimate = base + delta
        result[method] = {"rmsle": rmsle(actual, estimate), "3107_daytime": rmsle(actual[early], estimate[early]),
                          "week_gain_interval": interval(frame, actual, base, estimate),
                          "changed_rows": int(np.count_nonzero(np.abs(delta) > 1e-10))}
    # These held-out errors characterize the maximum available improvement and
    # are never used by adjustment().
    residual = (frame.y_log - base).where(early)
    level = residual.groupby(frame.date).transform("mean").where(early, 0.).fillna(0.)
    shape = residual.fillna(0.) - level
    group = [frame.date.dt.to_period("M"), weekday_group(pd.DatetimeIndex(frame.date)), frame.hour]
    recurring = shape.where(early).groupby(group).transform("mean").where(early, 0.).fillna(0.)
    result["oracle_gain_3107_daytime"] = {
        "perfect_date_level": baseline - rmsle(actual, base + level),
        "perfect_date_shape": baseline - rmsle(actual, base + shape),
        "perfect_recurring_month_weekday_hour_shape": baseline - rmsle(actual, base + recurring),
    }
    return result


def run(args):
    episodes, raw, inputs, protected, calibrated, fallback = load_episodes(args)
    fitted = trained_inputs(raw, args.work_dir, args.earlier, *protected)
    truth = raw.set_index(KEY).observed
    reports, cases = {}, {}
    for name, (rows, origin, _) in episodes.items():
        rows = rows.reset_index(drop=True)
        keys = pd.MultiIndex.from_frame(rows[KEY])
        if not keys.is_unique or not np.allclose(truth.reindex(keys), rows.y_log, atol=1e-8, rtol=0):
            raise ValueError(f"Archived labels do not match raw training data for {name}.")
        episode, _ = replay(rows, origin, raw, inputs, protected, calibrated, fallback, fitted)
        episode["combined"] = episode.combined.clip(lower=0.)
        changes, support = adjustment(raw.loc[raw.date.le(origin)], episode[[*KEY, "combined"]], origin, protected)
        cases[name] = episode[[*KEY, "combined", "y_log"]].assign(**changes)
        reports[name] = {"origin": str(origin.date()), **support, **score(cases[name])}
    cases["disjoint_A_C"] = pd.concat([cases["A"], cases["C"]], ignore_index=True)
    cases["disjoint_warm"] = pd.concat([cases[origin].loc[cases[origin].date.dt.month.eq(month)] for origin, month in
                                       (("2025-07-31", 8), ("2025-08-31", 9), ("2025-09-30", 10))], ignore_index=True)
    for name in ("disjoint_A_C", "disjoint_warm"):
        if cases[name].duplicated(KEY).any():
            raise ValueError(f"Duplicate validation keys in {name}.")
        reports[name] = score(cases[name])
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("team", "earlier", "recent", "extra_xgb", "june_xgb", "april_xgb"):
        parser.add_argument(argument, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    started = perf_counter()
    result = run(args)
    result["seconds"] = round(perf_counter() - started, 2)
    report = json.dumps(result, indent=2, allow_nan=False)
    (args.work_dir / "sincon_3107_mature_hour_profile_results.json").write_text(report + "\n")
    print(report)
