"""Recalibrate the 21–23h log profile while preserving its station-day mean."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sincon_complementary_blend import rmsle
from sincon_fuel_shock import read_archive
from sincon_late_low_count_calibration import uncertainty
from sincon_station_state import KEY, WINDOWS
from sincon_weekly_variability import protected_days


DAY = KEY[:2]
HOURS = [21, 22, 23]
AXES = np.array([[-1., 0., 1.], [1., -2., 1.]]) / np.sqrt([[2.], [6.]])
LOG_FIT = ["slope_shift", "curve_shift"]
LOG_ERROR = ["slope_error", "curve_error"]
MIN_GAIN = .0021


def panel(frame, columns, holidays, events):
    data = frame.loc[frame.hour.isin(HOURS)].pivot(index=DAY, columns="hour", values=columns)
    data = data.reindex(columns=pd.MultiIndex.from_product([columns, HOURS]))
    if data.isna().any().any():
        raise ValueError("Every station-day must have its three late observations or forecasts.")
    keys = data.index.to_frame(index=False)
    keys["ordinary"] = ~keys.date.isin(holidays) & ~data.index.isin(events)
    return keys, {name: data[name].to_numpy() for name in columns}


def weekday_groups(dow):
    return np.maximum(dow - 3, 0)


def intensity(values, dates, ordinary):
    calendar_days = dates.to_numpy(dtype="datetime64[D]").astype("int64")
    gaps = np.abs(calendar_days[:, None] - calendar_days[None, :])
    dow = dates.dt.dayofweek.to_numpy()
    group = weekday_groups(dow)
    comparable = ((dow[:, None] == dow[None, :]) & (gaps > 0) & (gaps <= 42) & ordinary[None, :])
    result = np.zeros(len(values))
    for i in np.flatnonzero(ordinary):
        use = comparable[i]
        if use.sum() < 3:
            use = (group == group[i]) & ordinary & (gaps[i] > 0) & (gaps[i] <= 42)
        if use.sum() >= 3:
            result[i] = values[i] - np.median(values[use])
    return np.clip(result, -.5, .5)


def proposals(train, train_values, frame, origin, holidays, events):
    target, values = panel(frame, ["local_stack"], holidays, events)
    if target.date.min() <= origin:
        raise ValueError("Late profiles must be forecast after their cutoff.")
    result = np.zeros((len(target), 2))
    for station, future_index in target.groupby("station_id").indices.items():
        previous_index = np.flatnonzero(train.station_id.eq(station) & train.date.le(origin) & train.ordinary)
        current_index = future_index[target.ordinary.to_numpy()[future_index]]
        if len(previous_index) < 20 or not len(current_index):
            continue
        history = train.iloc[previous_index]
        future = target.iloc[current_index]
        past_value = train_values[previous_index]
        base = values["local_stack"][current_index]
        historical_intensity = intensity(past_value.mean(axis=1), history.date, history.ordinary.to_numpy())
        predicted_intensity = intensity(base.mean(axis=1), future.date, future.ordinary.to_numpy())
        historical_weekday = history.date.dt.dayofweek.to_numpy()
        future_weekday = future.date.dt.dayofweek.to_numpy()
        match = future_weekday[:, None] == historical_weekday[None, :]
        peers = weekday_groups(future_weekday)[:, None] == weekday_groups(historical_weekday)[None, :]
        day_distance = np.abs(future.date.dt.dayofyear.to_numpy()[:, None] - history.date.dt.dayofyear.to_numpy()[None, :])
        day_distance = np.minimum(day_distance, 366 - day_distance)
        recency = np.exp2(-(origin - history.date).dt.days.to_numpy() / 150.)
        intensity_distance = np.abs(predicted_intensity[:, None] - historical_intensity[None, :])
        weights = recency[None, :] * (0.40 + .60 * np.exp(-day_distance / 35.)) * np.exp(-intensity_distance / .20)
        weights *= np.where(match, 1., np.where(peers, .35, 0.))
        mass = weights.sum(axis=1)
        effective = np.divide(mass ** 2, np.square(weights).sum(axis=1), out=np.zeros(len(mass)), where=mass > 0.)
        active = effective >= 4.
        shape = (weights[active] @ past_value) / mass[active, None]
        result[current_index[active]] = ((shape - base[active]) @ AXES.T).clip(-.5, .5)
    shift = pd.DataFrame(result, columns=LOG_FIT)
    dow = target.date.dt.dayofweek.to_numpy()
    return pd.concat([target, shift], axis=1).assign(day_group=weekday_groups(dow), origin=origin), values


def observed_errors(frame, target, values):
    actual = frame.loc[frame.hour.isin(HOURS)].pivot(index=DAY, columns="hour", values="y_log")
    actual = actual.reindex(pd.MultiIndex.from_frame(target[DAY])).reindex(columns=HOURS).to_numpy()
    if not np.isfinite(actual).all():
        raise ValueError("Archived labels must cover every station-day forecast.")
    return actual - np.maximum(values["local_stack"], 0.)


def build_history(observed, train_values, archive, holidays, events):
    panels, history = {}, []
    for origin, frame in archive.groupby("origin", sort=True):
        target, values = proposals(observed, train_values, frame, origin, holidays, events)
        panels[origin] = target, values
        residual = pd.DataFrame(observed_errors(frame, target, values) @ AXES.T, columns=LOG_ERROR)
        history.append(pd.concat([target, residual], axis=1))
    return panels, pd.concat(history, ignore_index=True)


def weights(past, future, origin):
    history = past.loc[past.origin.lt(origin) & past.date.le(origin)].sort_values("origin").drop_duplicates(DAY, keep="last")
    predictors = history[LOG_FIT].to_numpy()
    usable = np.abs(predictors).max(axis=1) > 1e-8 if len(history) else np.zeros(0, dtype=bool)
    history, predictors = history.loc[usable], predictors[usable]
    if history.date.nunique() < 30:
        return np.full((len(future), 2), .20), int(history.date.nunique())
    response = history[LOG_ERROR].to_numpy().clip(-.5, .5)
    dot = predictors * response
    square = predictors ** 2
    prior = (dot.sum(axis=0) + 10. * .20) / (square.sum(axis=0) + 10.)
    group_key = pd.Index(history.day_group, name="day_group")
    group_dot = pd.DataFrame(dot, index=group_key).groupby(level=0).sum()
    group_square = pd.DataFrame(square, index=group_key).groupby(level=0).sum()
    group = (group_dot + 12. * prior) / (group_square + 12.)
    station_key = pd.MultiIndex.from_frame(history[["station_id", "day_group"]])
    local_dot = pd.DataFrame(dot, index=station_key).groupby(level=[0, 1]).sum()
    local_square = pd.DataFrame(square, index=station_key).groupby(level=[0, 1]).sum()
    shrink = 8. * group.reindex(local_dot.index.get_level_values("day_group")).set_axis(local_dot.index)
    local = (local_dot + shrink) / (local_square + 8.)
    fallback = group.reindex(future.day_group).fillna(pd.Series(prior)).to_numpy()
    choices = local.reindex(pd.MultiIndex.from_frame(future[["station_id", "day_group"]])).to_numpy()
    return np.where(np.isfinite(choices), choices, fallback).clip(-.30, .45), int(history.date.nunique())


def apply(frame, baseline, target, values, coefficient):
    delta = (target[LOG_FIT].to_numpy() * coefficient) @ AXES
    base = np.maximum(values["local_stack"], 0.)
    maximum = np.abs(delta).max(axis=1)
    row_scale = np.minimum(1., .20 / np.maximum(maximum, 1e-8))
    legal = np.divide(base, np.maximum(-delta, 1e-8), out=np.ones_like(delta), where=delta < 0)
    delta *= np.minimum(row_scale, legal.min(axis=1))[:, None]
    late = frame.hour.isin(HOURS).to_numpy()
    index = pd.MultiIndex.from_frame(target[DAY])
    rows = index.get_indexer(pd.MultiIndex.from_frame(frame.loc[late, DAY]))
    if (rows < 0).any() or np.abs(delta.sum(axis=1)).max() > 1e-10:
        raise ValueError("Changes must preserve the station-day late log level and align to their source forecasts.")
    result = baseline.copy()
    result[late] += delta[rows, frame.loc[late, "hour"].to_numpy() - HOURS[0]]
    return result


def metrics(frame, forecast, eligible):
    rows = frame.loc[eligible]
    actual = rows.y_log.to_numpy()
    before = rows.local_stack.to_numpy()
    after = forecast[eligible]
    late = rows.hour.isin(HOURS).to_numpy()
    first = rmsle(actual, before)
    second = rmsle(actual, after)
    return {"baseline": first, "candidate": second, "gain": first - second,
            "late_baseline": rmsle(actual[late], before[late]), "late_candidate": rmsle(actual[late], after[late]),
            "week_block_gain_ci95": uncertainty(rows.date, actual, before, after, 7)}


def run(work, team, earlier, recent=None):
    started = perf_counter()
    holidays, events = protected_days(work)
    train = pd.read_csv(team / "data/train.csv", usecols=[*KEY, "boardings"], parse_dates=["date"])
    observed, train_values = panel(train.assign(log=np.log1p(train.boardings)), ["log"], holidays, events)
    current, archive = read_archive(work / "sincon_local_validation.csv.gz", earlier)
    panels, history = build_history(observed, train_values["log"], archive, holidays, events)
    summer = pd.read_csv(work / "sincon_summer_replay_validation.csv.gz", usecols=[*KEY, "y_log", "local_stack"], parse_dates=["date"])
    cases = {name: (pd.Timestamp(cutoff), current.loc[current.fold.eq(name)], None) for name, (cutoff, _) in WINDOWS.items()}
    cases["summer_2025_30_seed"] = (pd.Timestamp("2025-06-30"), summer, None)
    aug = pd.Timestamp("2025-07-31")
    cases["august_2025_rolling"] = (aug, archive.loc[archive.origin.eq(aug)], pd.Timestamp("2025-08-31"))
    if recent is not None:
        for path in sorted(recent.glob("origin_*.csv.gz")):
            frame = pd.read_csv(path, usecols=[*KEY, "origin", "y_log", "local_stack"], parse_dates=["date", "origin"])
            if frame.origin.nunique() != 1:
                raise ValueError("Each recent archive must have one forecast origin.")
            cases[path.name.removesuffix(".csv.gz")] = (frame.origin.iloc[0], frame, None)
    result = {"minimum_gain": MIN_GAIN, "cases": {}}
    for name, (origin, frame, end) in cases.items():
        target, values = (panels[origin] if origin in panels and name != "summer_2025_30_seed"
                          else proposals(observed, train_values["log"], frame, origin, holidays, events))
        historical_weight, fit_days = weights(history, target, origin)
        baseline = frame.local_stack.to_numpy()
        estimates = {"fixed_20_percent": apply(frame, baseline, target, values, .20),
                     "preorigin_learned": apply(frame, baseline, target, values, historical_weight)}
        available = np.ones(len(frame), dtype=bool) if end is None else frame.date.le(end).to_numpy()
        mature = available & ~frame.station_id.isin([3107, 3108, 3109]).to_numpy()
        actual_late_error = observed_errors(frame, target, values)
        supported_days = target.ordinary.to_numpy() if end is None else (target.ordinary & target.date.le(end)).to_numpy()
        shape = actual_late_error[supported_days] @ AXES.T
        shape_sse = float(np.square(shape).sum())
        overall_sse = float(np.square(frame.y_log.to_numpy()[available] - np.maximum(baseline[available], 0.)).sum())
        before = np.sqrt(overall_sse / available.sum())
        # Diagnostic only: validation truth establishes the upper bound for a
        # fixed station-and-weekday adjustment. The prediction never uses it.
        weekday = target.loc[supported_days, "date"].dt.dayofweek.to_numpy()
        station = target.loc[supported_days, "station_id"].to_numpy()
        static_sse = float(np.square(pd.DataFrame(shape).groupby([station, weekday]).transform("mean").to_numpy()).sum())
        bounds = {"station_weekday": float(before - np.sqrt((overall_sse - static_sse) / available.sum())),
                  "all_ordinary_late_shape": float(before - np.sqrt((overall_sse - shape_sse) / available.sum()))}
        result["cases"][name] = {"origin": str(origin.date()), "fit_days": fit_days, "ordinary_late_shape_sse_share": shape_sse / overall_sse,
                                 "diagnostic_max_gain": bounds,
                                 "estimated_weights_mean": historical_weight.mean(axis=0).tolist(),
                                 "scores": {variant: {"all": metrics(frame, forecast, available), "mature": metrics(frame, forecast, mature)}
                                            for variant, forecast in estimates.items()}}
    result["elapsed_seconds"] = round(perf_counter() - started, 2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team", type=Path)
    parser.add_argument("earlier", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--recent", type=Path)
    args = parser.parse_args()
    output = json.dumps(run(args.work_dir, args.team, args.earlier, args.recent), indent=2, allow_nan=False) + "\n"
    (args.work_dir / "sincon_late_composition_kernel_results.json").write_text(output)
