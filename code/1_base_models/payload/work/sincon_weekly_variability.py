"""Calibrate the strength of short-lived weekly variation in prior forecasts."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sincon_station_state import KEY, WINDOWS


DAY = KEY[:2]
LEVEL = ["station_id", "weekend"]
SHAPE = ["station_id", "weekend", "hour"]
HOLIDAY_FLAGS = ["is_holiday", "is_bridge_day", "is_quasi_holiday", "is_pre_long_offday", "is_post_long_offday"]
MINIMUM_GAIN = .003


def read_csv(path, columns):
    return pd.read_csv(path, usecols=columns, parse_dates=["date"])


def protected_days(work):
    with ZipFile(work / "sincon_local_repro.zip") as archive:
        root = "external_inputs/"
        calendar = read_csv(archive.open(root + "calendar_features.csv"), ["date", *HOLIDAY_FLAGS])
        events = read_csv(archive.open(root + "external/events_daily.csv"), DAY)
    festivals = read_csv(work / "sincon_festival_station_days.csv", DAY)
    baseball = read_csv(work / "sincon_kbo_game_data.csv", ["date"]).assign(station_id=3127)
    football = read_csv(work / "sincon_bucheon_football_data.csv", ["date"]).assign(station_id=3754)
    event_days = pd.MultiIndex.from_frame(pd.concat([events, festivals, baseball, football]).drop_duplicates())
    return pd.DatetimeIndex(calendar.loc[calendar[HOLIDAY_FLAGS].ne(0).any(axis=1), "date"]), event_days


def weekly_signal(frame, holidays, events):
    forecast = frame[[*KEY, "base"]].sort_values(KEY).reset_index(drop=True)
    days = pd.DatetimeIndex(forecast.date.unique())
    stations = pd.Index(forecast.station_id.unique())
    grid = pd.MultiIndex.from_product([days, stations, range(6, 24)], names=KEY)
    if not grid.equals(pd.MultiIndex.from_frame(forecast[KEY])) or (days[1:] - days[:-1]).days.max() != 1:
        raise ValueError("Weekly calibration requires the full, consecutive station-hour prediction grid.")

    values = forecast.base.to_numpy().reshape(len(days), len(stations), 18)
    station_days = pd.MultiIndex.from_product([days, stations], names=DAY)
    ordinary = (~station_days.isin(events)).reshape(len(days), len(stations)) & ~days.isin(holidays)[:, None]
    sample = np.where(ordinary[..., None], values, np.nan)
    average = np.zeros_like(values)
    weights = np.zeros(values.shape[:2])
    neighbors = np.zeros(values.shape[:2], dtype=int)
    for offset, weight in ((-21, .5), (-14, .7), (-7, 1.), (7, 1.), (14, .7), (21, .5)):
        positions = np.arange(len(days)) + offset
        shifted = sample[np.clip(positions, 0, len(days) - 1)]
        valid = (positions >= 0)[:, None] & (positions < len(days))[:, None] & np.isfinite(shifted).all(axis=2)
        average += np.where(valid[..., None], shifted * weight, 0.)
        weights += valid * weight
        neighbors += valid
    eligible = ordinary & (neighbors >= 2)
    smooth = np.divide(average, weights[..., None], out=np.zeros_like(average), where=eligible[..., None])
    departure = np.where(eligible[..., None], values - smooth, 0.)
    level = departure.mean(axis=2, keepdims=True)
    return forecast[KEY].assign(
        weekend=forecast.date.dt.dayofweek.ge(5),
        level=np.broadcast_to(level, departure.shape).ravel(),
        shape=(departure - level).ravel(),
        eligible=np.broadcast_to(eligible[..., None], departure.shape).ravel(),
    )


def past_episodes(validation, earlier, holidays, events):
    old = []
    for path in sorted(earlier.glob("origin_*.csv.gz")):
        cutoff = pd.Timestamp(path.name[7:17])
        sample = read_csv(path, [*KEY, "y_log", "local_stack"]).rename(columns={"local_stack": "base"})
        old.append((cutoff, sample))
    current = read_csv(validation, [*KEY, "fold", "y_log", "local_stack"]).rename(columns={"local_stack": "base"})
    current = {fold: current.loc[current.fold.eq(fold), [*KEY, "base", "y_log"]].copy() for fold in WINDOWS}
    archives = [*old, *((pd.Timestamp(WINDOWS[fold][0]), rows) for fold, rows in current.items())]
    signals = []
    for origin, rows in archives:
        if rows.date.min() <= origin:
            raise ValueError("An archived forecast overlaps its own training period.")
        features = weekly_signal(rows, holidays, events)
        targets = rows.assign(error=rows.y_log - rows.base)[[*KEY, "error"]]
        signals.append(features.merge(targets, on=KEY, validate="one_to_one").assign(origin=origin))
    return current, pd.concat(signals, ignore_index=True)


def slope(fit, target, signal, error, groups, common, penalty):
    stats = fit.assign(dot=fit[signal] * fit[error], square=fit[signal] ** 2)
    pooled = stats.groupby(common)[["dot", "square"]].sum()
    pooled = pooled["dot"] / (pooled["square"] + 1.5)
    prior = target[common].join(pooled.rename("prior"), on=common, validate="many_to_one").prior.fillna(0.)
    by_group = stats.groupby(groups)[["dot", "square"]].sum()
    local = target[groups].join(by_group, on=groups, validate="many_to_one").fillna(0.)
    return ((local["dot"] + penalty * prior) / (local["square"] + penalty)).clip(-.8, .35)


def correction(past, future, origin):
    usable = past.loc[past.origin.lt(origin) & past.date.le(origin)]
    usable = usable.sort_values("origin").drop_duplicates(KEY, keep="last")
    fit = usable.loc[usable.eligible].copy()
    if fit.date.nunique() < 75 or future.date.min() <= origin:
        raise ValueError("Weekly variability fit needs earlier errors and future prediction keys.")
    fit["level_error"] = fit.groupby(DAY).error.transform("mean").clip(-.45, .45)
    fit["shape_error"] = (fit.error - fit.groupby(DAY).error.transform("mean")).clip(-.55, .55)
    daily_fit = fit.drop_duplicates(DAY)
    level = slope(daily_fit, future, "level", "level_error", LEVEL, ["weekend"], .4) * future.level
    shape = slope(fit, future, "shape", "shape_error", SHAPE, ["weekend", "hour"], .35) * future["shape"]
    shape -= shape.groupby([future.date, future.station_id]).transform("mean")
    # Keep the existing station-level calibration and apply only date and
    # hour contrasts on the eligible part of the prediction grid.
    delta = level + shape
    center = delta.where(future.eligible).groupby(future.station_id).transform("mean").fillna(0)
    adjusted = .65 * (delta - center).where(future.eligible, 0.)
    return adjusted, int(fit.date.nunique())


def rmsle(rows, prediction):
    return float(np.sqrt(np.square(rows.y_log - prediction.clip(lower=0)).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("earlier", type=Path)
    parser.add_argument("team_dir", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    work = args.work_dir
    holidays, events = protected_days(work)
    current, past = past_episodes(work / "sincon_local_validation.csv.gz", args.earlier, holidays, events)
    report = {"folds": {}}
    for fold, (cutoff, _) in WINDOWS.items():
        start = perf_counter()
        future = weekly_signal(current[fold], holidays, events).merge(current[fold], on=KEY, validate="one_to_one")
        delta, days = correction(past, future, pd.Timestamp(cutoff))
        late = future.hour.ge(21)
        overall = {"baseline": rmsle(future, future.base), "weekly": rmsle(future, future.base + delta)}
        report["folds"][fold] = {
            "overall": overall,
            "late": {"baseline": rmsle(future[late], future.base[late]), "weekly": rmsle(future[late], (future.base + delta)[late])},
            "eligible_rows": int(future.eligible.sum()), "training_days": days,
            "mean_absolute_log_change": float(delta.abs().mean()), "seconds": round(perf_counter() - start, 2),
        }
        print(json.dumps({fold: report["folds"][fold]}, allow_nan=False), flush=True)
    report["accepted"] = sum(
        f["overall"]["baseline"] - f["overall"]["weekly"] >= MINIMUM_GAIN for f in report["folds"].values()
    ) >= 2
    if report["accepted"]:
        test = read_csv(args.team_dir / "data/test.csv", [*KEY, "example_id"])
        base = pd.read_csv(args.team_dir / "submission_local_stack_level_station.csv", float_precision="round_trip")
        if len(test) != 157_194 or not test.example_id.is_unique or not test.example_id.equals(base.example_id):
            raise ValueError("The calibrated CSV must match the test keys.")
        future = weekly_signal(test[KEY].assign(base=np.log1p(base.boardings)), holidays, events)
        delta, _ = correction(past, future, pd.Timestamp("2026-06-30"))
        change = future[KEY].assign(delta=delta).set_index(KEY).delta.reindex(pd.MultiIndex.from_frame(test[KEY]))
        forecast = np.expm1((np.log1p(base.boardings) + change.to_numpy()).clip(lower=0))
        if not np.isfinite(forecast).all() or (forecast < 0).any():
            raise ValueError("Submission values must be finite and nonnegative.")
        path = work / "submission_local_stack_level_station_weekly.csv"
        temporary = path.with_suffix(".tmp")
        base.assign(boardings=forecast).to_csv(temporary, index=False)
        temporary.replace(path)
        report["submission"] = str(path.resolve())
    (work / "sincon_weekly_variability_results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
