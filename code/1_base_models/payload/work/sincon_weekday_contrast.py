"""Use ordinary-week ridership to regularize archived within-day forecasts."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sincon_station_state import KEY, WINDOWS


HOURS = np.arange(6, 24)
DAY = KEY[:2]
CALENDAR_EXCLUSIONS = [
    "is_holiday", "is_lunar_season", "is_bridge_day", "is_quasi_holiday",
    "is_pre_long_offday", "is_post_long_offday",
]
BLEND = 0.5
HALF_LIFE = 112
ADOPTION_GAIN = 0.003


def ordinary_dates_and_events(work):
    with ZipFile(work / "sincon_local_repro.zip") as archive:
        calendar = pd.read_csv(archive.open("external_inputs/calendar_features.csv"), parse_dates=["date"])
        events = pd.read_csv(archive.open("external_inputs/external/events_daily.csv"), parse_dates=["date"])[DAY]
    festival = pd.read_csv(work / "sincon_festival_station_days.csv", usecols=DAY, parse_dates=["date"])
    sports = [
        pd.read_csv(work / name, usecols=["date"], parse_dates=["date"]).assign(station_id=station)
        for name, station in (("sincon_kbo_game_data.csv", 3127), ("sincon_bucheon_football_data.csv", 3754))
    ]
    ordinary = calendar.loc[calendar[CALENDAR_EXCLUSIONS].eq(0).all(axis=1), "date"]
    return pd.DatetimeIndex(ordinary), pd.MultiIndex.from_frame(pd.concat([events, festival, *sports]).drop_duplicates())


def panel(frame, value, stations, ordinary, events):
    dates = pd.date_range(frame.date.min(), frame.date.max())
    keys = pd.MultiIndex.from_product([dates, stations, HOURS], names=KEY)
    values = frame.set_index(KEY)[value].reindex(keys).to_numpy().reshape(len(dates), len(stations), len(HOURS))
    station_days = keys.droplevel("hour").unique()
    regular = dates.isin(ordinary)[:, None] & ~station_days.isin(events).reshape(len(dates), len(stations))
    return dates, values, regular


def week_shapes(dates, values, ordinary):
    mondays = dates[(dates.dayofweek == 0) & (dates <= dates[-1] - pd.Timedelta(days=6))]
    week_days = dates.get_indexer((mondays.to_numpy()[:, None] + np.arange(7).astype("timedelta64[D]")).ravel())
    week_days = week_days.reshape(len(mondays), 7)
    complete = (ordinary[week_days] & np.isfinite(values[week_days]).all(axis=3)).all(axis=1)
    daily = values[week_days]
    centered = daily - daily.mean(axis=3, keepdims=True)
    return mondays, np.where(complete[:, None, :, None], centered, np.nan), complete


def profiles(weeks, complete, weights):
    weight = (weights[:, None] * complete)[:, None, :, None]
    valid = np.nan_to_num(weeks)
    mass = weight.sum(axis=0)
    mean = np.divide((weight * valid).sum(axis=0), mass, out=np.zeros(weeks.shape[1:]), where=mass > 0)
    variance = np.divide(
        (weight * np.square(valid - mean)).sum(axis=0), mass, out=np.zeros_like(mean), where=mass > 0,
    )
    n_eff = np.divide(mass ** 2, (weight ** 2).sum(axis=0), out=np.zeros_like(mass), where=mass > 0)
    return mean, variance, n_eff


def pool_interactions(observed, variance, n_eff, station_line):
    interaction = observed - observed.mean(axis=0, keepdims=True)
    pooled = np.zeros_like(interaction)
    for line in np.unique(station_line):
        row = np.flatnonzero((station_line == line) & (n_eff[0, :, 0] >= 4))
        if len(row) == 0:
            continue
        prior = interaction[:, row].mean(axis=1, keepdims=True)
        sampling_variance = variance[:, row] / n_eff[:, row]
        signal = np.maximum(np.var(interaction[:, row], axis=1, keepdims=True) -
                            sampling_variance.mean(axis=1, keepdims=True), 0.001)
        reliability = signal / (signal + sampling_variance)
        pooled[:, row] = prior + reliability * (interaction[:, row] - prior)
    return pooled


def forecast_adjustment(history, future, cutoff, metadata, ordinary, events):
    if history.date.max() > cutoff or future.date.min() <= cutoff or "y_log" in future:
        raise ValueError("Fit expects pre-origin observations and unlabeled forecast rows.")
    stations = metadata.index
    hdates, hvalues, hordinary = panel(history, "log", stations, ordinary, events)
    fdates, fvalues, fordinary = panel(future, "base", stations, ordinary, events)
    hw, hshape, hcomplete = week_shapes(hdates, hvalues, hordinary)
    fw, fshape, fcomplete = week_shapes(fdates, fvalues, fordinary)
    weights = np.exp2(-(cutoff - hw).days.to_numpy() / HALF_LIFE)
    observed, variance, n_eff = profiles(hshape, hcomplete, weights)
    predicted, _, n_forecast = profiles(fshape, fcomplete, np.ones(len(fw)))
    stable = (n_eff >= 4) & (n_forecast >= 2)
    interaction = pool_interactions(observed, variance, n_eff, metadata.canonical_line.to_numpy())
    forecast_interaction = predicted - predicted.mean(axis=0, keepdims=True)
    station_hour = observed.mean(axis=0, keepdims=True) - predicted.mean(axis=0, keepdims=True)
    result = {}
    all_keys = pd.MultiIndex.from_product([fdates, stations, HOURS], names=KEY)
    target = pd.MultiIndex.from_frame(future[KEY])
    for name, proposal in {
        "weekday_interaction": interaction - forecast_interaction,
        "weekday_and_hour": interaction - forecast_interaction + station_hour,
    }.items():
        delta = np.where(stable, proposal.clip(-0.35, 0.35), 0)
        delta -= delta.mean(axis=2, keepdims=True)
        daily = BLEND * np.where(fordinary[:, :, None], delta[fdates.dayofweek], 0)
        result[name] = pd.Series(daily.ravel(), index=all_keys).reindex(target).to_numpy()
    return result, int(np.count_nonzero(stable))


def score(actual, prediction, late):
    error = np.square(actual - np.maximum(prediction, 0))
    return {"overall": float(np.sqrt(error.mean())), "late": float(np.sqrt(error[late].mean()))}


def run(team, work):
    history = pd.read_csv(team / "data/train.csv", usecols=[*KEY, "boardings"], parse_dates=["date"])
    history["log"] = np.log1p(history.pop("boardings"))
    metadata = pd.read_csv(team / "data/metaData.csv", usecols=["station_id", "canonical_line"]).set_index("station_id")
    archive = pd.read_csv(work / "sincon_local_validation.csv.gz", usecols=[*KEY, "fold", "y_log", "local_stack"],
                          parse_dates=["date"])
    ordinary, events = ordinary_dates_and_events(work)
    report = {"blend": BLEND, "half_life_days": HALF_LIFE, "folds": {}}
    for fold, (origin, _) in WINDOWS.items():
        start = perf_counter()
        cutoff = pd.Timestamp(origin)
        rows = archive.loc[archive.fold.eq(fold)]
        base, actual = rows.local_stack.to_numpy(), rows.y_log.to_numpy()
        truth = history.set_index(KEY).log.reindex(pd.MultiIndex.from_frame(rows[KEY]))
        if not np.allclose(truth, actual, atol=1e-8, rtol=0):
            raise ValueError(f"Archived labels do not match the training records for fold {fold}.")
        earlier = history.loc[history.date.le(cutoff)]
        future = rows[KEY].assign(base=base)
        adjustments, support = forecast_adjustment(earlier, future, cutoff, metadata, ordinary, events)
        for delta in adjustments.values():
            shift = future[DAY].assign(delta=delta).groupby(DAY).delta.mean()
            if not np.isfinite(delta).all() or shift.abs().max() > 1e-12:
                raise ValueError(f"Adjustment changed station-day levels for fold {fold}.")
        late = rows.hour.ge(21).to_numpy()
        report["folds"][fold] = {
            "reference": score(actual, base, late),
            **{name: score(actual, base + delta, late) for name, delta in adjustments.items()},
            "supported_stations": support,
            "seconds": round(perf_counter() - start, 2),
        }
    report["adoption"] = {
        name: [fold for fold, result in report["folds"].items()
               if result["reference"]["overall"] - result[name]["overall"] >= ADOPTION_GAIN]
        for name in adjustments
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    results = json.dumps(run(args.team, args.work_dir), indent=2, allow_nan=False)
    (args.work_dir / "sincon_weekday_contrast_results.json").write_text(results + "\n")
    print(results, flush=True)
