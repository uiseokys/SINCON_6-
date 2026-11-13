"""Test whether known fuel price changes explain archived forecast errors."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sincon_station_state import KEY, WINDOWS


PRICE = ["short_change", "regional_spread"]
HOLIDAY = ["is_holiday", "is_bridge_day", "is_quasi_holiday", "is_pre_long_offday", "is_post_long_offday"]
WEIGHT = 0.5


def fuel_prices(path):
    raw = pd.read_csv(path, encoding="cp949")
    dates = pd.to_datetime(raw.pop("구분"), format="%Y년%m월%d일")
    price = raw.set_index(dates).sort_index().rename_axis("date")
    if price.index.has_duplicates or not price.index.equals(pd.date_range(price.index.min(), price.index.max())):
        raise ValueError("Fuel prices must have one observation per date.")
    if price[["인천", "경기"]].isna().any().any() or price[["인천", "경기"]].le(0).any().any():
        raise ValueError("Positive observed Incheon and Gyeonggi fuel prices are required.")
    local = np.log(price[["인천", "경기"]]).mean(axis=1)
    other = np.log(price.drop(columns=["서울", "인천", "경기"])).median(axis=1)
    current = local.rolling(7).mean()
    return pd.DataFrame({
        "short_change": current - local.shift(7).rolling(28).mean(),
        "regional_spread": (local - other).rolling(7).mean(),
    })


def read_archive(path, earlier):
    old = [
        pd.read_csv(source, usecols=[*KEY, "origin", "y_log", "local_stack"], parse_dates=["date", "origin"])
        for source in sorted(earlier.glob("origin_*.csv.gz"))
    ]
    current = pd.read_csv(path, usecols=[*KEY, "fold", "y_log", "local_stack"], parse_dates=["date"])
    archive = pd.concat([*old, current.assign(origin=pd.to_datetime(current.fold.map({k: v[0] for k, v in WINDOWS.items()})))],
                        ignore_index=True)
    if archive.origin.ge(archive.date).any() or archive.duplicated(["origin", *KEY]).any():
        raise ValueError("Each archived forecast must precede its target and have unique keys.")
    return current, archive


def daily_context(work, prices):
    with ZipFile(work / "sincon_local_repro.zip") as bundle:
        calendar = pd.read_csv(bundle.open("external_inputs/calendar_features.csv"), parse_dates=["date"])
        weather = pd.read_csv(bundle.open("external_inputs/external/weather_hourly.csv"), usecols=[
            "date", "weather_zone", "hour", "day_precip", "day_temp_mean",
        ], parse_dates=["date"])
    weather = weather.loc[weather.hour.eq(12)].groupby("date")[["day_precip", "day_temp_mean"]].mean()
    context = calendar.set_index("date").join(weather, validate="one_to_one").join(prices, validate="one_to_one")
    context["ordinary"] = context[HOLIDAY].eq(0).all(axis=1)
    return context


def band(hours):
    return np.searchsorted([10, 16, 21], hours, side="right")


def standardized(fit, target, fields):
    center = fit[fields].mean()
    scale = fit[fields].std().clip(lower=1e-4)
    a = ((fit[fields] - center) / scale).to_numpy()
    b = ((target[fields] - center) / scale).to_numpy()
    return a, np.clip(b, np.quantile(a, .025, axis=0), np.quantile(a, .975, axis=0))


def design(frame, weather):
    annual = frame.index.dayofyear * 2 * np.pi / 365.2425
    weekday = np.eye(7)[frame.index.dayofweek]
    return np.column_stack([weekday, np.sin(annual), np.cos(annual), weather])


def estimate(archive, target, context, cutoff):
    history = archive.loc[archive.date.le(cutoff) & archive.origin.lt(cutoff)]
    history = history.sort_values("origin").drop_duplicates(KEY, keep="last")
    history = history.assign(band=band(history.hour), error=history.y_log - history.local_stack.clip(lower=0))
    errors = history.groupby(["date", "band"]).error.mean().unstack("band")
    fits = context.reindex(errors.index)
    keep = fits.ordinary & fits[PRICE].notna().all(axis=1)
    fits, errors = fits.loc[keep], errors.loc[keep]
    future = context.reindex(pd.DatetimeIndex(target.date.unique()).sort_values())
    if len(fits) < 60 or future[PRICE].isna().any().any() or future.index.min() <= cutoff:
        raise ValueError("Fuel correction needs complete future prices and sufficient earlier archived errors.")

    weather_a, weather_b = standardized(fits, future, ["day_precip", "day_temp_mean"])
    price_a, price_b = standardized(fits, future, PRICE)
    x = design(fits, weather_a)
    z = design(future, weather_b)
    # Isolate price variation beyond calendar and weather, using only
    # historical dates to fit the price residualization and error response.
    control = x.T @ x + np.diag([1.] * 7 + [15., 15., 10., 10.])
    relation = np.linalg.solve(control, x.T @ price_a)
    a = price_a - x @ relation
    b = price_b - z @ relation
    response = errors.to_numpy().clip(-.18, .18)
    baseline = x @ np.linalg.solve(control, x.T @ response)
    beta = np.linalg.solve(a.T @ a + 45 * np.eye(len(PRICE)), a.T @ (response - baseline))
    delta = (b @ beta).clip(-.06, .06)
    # Preserve the prior full-window level and keep special days with the
    # current model, which has a separate response to holidays and events.
    ordinary = future.ordinary.to_numpy()
    delta -= delta[ordinary].mean(axis=0, keepdims=True)
    delta = pd.DataFrame(np.where(ordinary[:, None], delta, 0), index=future.index)
    row_dates = delta.index.get_indexer(target.date)
    return delta.to_numpy()[row_dates, band(target.hour)], {
        "training_days": len(fits), "last_training_date": fits.index.max().strftime("%Y-%m-%d"),
        "training_price_change_range": [float(fits.short_change.min()), float(fits.short_change.max())],
        "target_price_change_range": [float(future.short_change.min()), float(future.short_change.max())],
    }


def rmsle(errors):
    return float(np.sqrt(np.mean(np.square(errors))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prices", type=Path)
    parser.add_argument("earlier", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    work = args.work_dir
    current, archive = read_archive(work / "sincon_local_validation.csv.gz", args.earlier)
    context = daily_context(work, fuel_prices(args.prices))
    results = {}
    for name, (origin, _) in WINDOWS.items():
        started = perf_counter()
        target = current.loc[current.fold.eq(name)].reset_index(drop=True)
        delta, details = estimate(archive, target[KEY], context, pd.Timestamp(origin))
        if not np.isfinite(delta).all():
            raise ValueError("Fuel corrections must be finite.")
        base = target.local_stack.clip(lower=0)
        before = target.y_log - base
        after = target.y_log - (base + WEIGHT * delta).clip(lower=0)
        late = target.hour.ge(21)
        results[name] = {
            "overall": {"baseline": rmsle(before), "fuel": rmsle(after)},
            "late": {"baseline": rmsle(before[late]), "fuel": rmsle(after[late])},
            "max_absolute_log_change": float(np.max(np.abs(WEIGHT * delta))),
            "seconds": round(perf_counter() - started, 2),
            **details,
        }
        print(json.dumps({name: results[name]}, allow_nan=False), flush=True)
    results["accepted"] = sum(scores["overall"]["baseline"] - scores["overall"]["fuel"] >= .003
                              for scores in results.values()) >= 2
    (work / "sincon_fuel_shock_results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
