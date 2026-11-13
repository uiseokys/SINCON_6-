"""Compare observed and modeled weather responses using matched weekday pairs."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sincon_station_state import KEY, WINDOWS
from sincon_team_bias_proxy import earlier_team_errors
from sincon_weekday_contrast import ordinary_dates_and_events, panel


HOURS = np.arange(6, 24)
TIME_BANDS = np.searchsorted([9, 13, 16, 19, 21], HOURS, side="right")
FEATURES = ["temperature", "heat", "rain", "rain_before", "rain_after", "daily_rain", "wind", "snow"]


def context(work, team):
    station = pd.read_csv(team / "data/metaData.csv", parse_dates=["first_date"]).set_index("station_id").sort_index()
    with ZipFile(work / "sincon_local_repro.zip") as archive:
        directory = "external_inputs/"
        zones = pd.read_csv(archive.open(directory + "external/weather_zone_map.csv")).set_index("station_id")
        calendar = pd.read_csv(archive.open(directory + "calendar_features.csv"), parse_dates=["date"]).set_index("date")
        weather = pd.read_csv(archive.open(directory + "external/weather_hourly.csv"), parse_dates=["date"])
    weather = weather.sort_values(["weather_zone", "date", "hour"])
    time = weather.date + pd.to_timedelta(weather.hour, unit="h")
    if not time.groupby(weather.weather_zone).diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Weather inputs must have a continuous hourly index in each zone.")
    by_zone = weather.groupby("weather_zone").precip
    before = by_zone.shift(1) + by_zone.shift(2)
    after = by_zone.shift(-1) + by_zone.shift(-2)
    weather = weather.assign(
        temperature=weather.temp_feel / 10, heat=(weather.temp_feel - 28).clip(lower=0) / 8,
        rain=np.log1p(4 * weather.precip), rain_before=np.log1p(before.fillna(0)),
        rain_after=np.log1p(after.fillna(0)), daily_rain=np.log1p(weather.day_precip) / 2,
        wind=weather.wind / 10, snow=np.log1p(weather.snowfall),
    ).set_index(["date", "weather_zone", "hour"])[FEATURES]
    ordinary, events = ordinary_dates_and_events(work)
    station = station.join(zones, validate="one_to_one")
    return station, calendar.school_vac_share, weather, ordinary, events


def weather_panel(dates, zone, weather):
    index = pd.MultiIndex.from_product([dates, zone, HOURS], names=weather.index.names)
    values = weather.reindex(index).to_numpy().reshape(len(dates), len(zone), len(HOURS), len(FEATURES))
    if not np.isfinite(values).all():
        raise ValueError("Weather observations must cover all selected date-zone-hours.")
    return values


def pairs(dates, values, valid, conditions, school):
    differences, design, usable, centers = [], [], [], []
    vacation = school.reindex(dates).to_numpy()
    for lag in (7, 14):
        keep = valid[lag:] & valid[:-lag] & (np.abs(vacation[lag:] - vacation[:-lag]) <= .18)[:, None]
        differences.append((values[lag:] - values[:-lag]).clip(-.55, .55))
        design.append((conditions[lag:] - conditions[:-lag]).clip(-2.5, 2.5))
        usable.append(keep[..., None] & np.isfinite(differences[-1]))
        centers.append(dates[lag:])
    return np.concatenate(differences), np.concatenate(design), np.concatenate(usable), centers[0].append(centers[1])


def response(data, station, reference_center, origin=None):
    y, x, mask, dates = data
    day_distance = dates.dayofyear.to_numpy() - reference_center.dayofyear
    weight = np.exp(-2 * np.sin(np.pi * day_distance / 365.25) ** 2)
    if origin is not None:
        weight *= np.exp2(-(origin - dates).days.to_numpy() / 240.)
    output = np.zeros((2, len(station), len(HOURS), len(FEATURES)))
    support = np.zeros((2, len(station), len(HOURS), len(FEATURES)))
    for weekend in (0, 1):
        active = (dates.dayofweek >= 5) == weekend
        weighted = mask[active] * weight[active, None, None]
        design = x[active]
        target = np.nan_to_num(y[active])
        gram = np.einsum("dshk,dshl,dsh->shkl", design, design, weighted, optimize=True)
        cross = np.einsum("dshk,dsh,dsh->shk", design, target, weighted, optimize=True)
        for line in station.canonical_line.unique():
            ids = np.flatnonzero(station.canonical_line.eq(line).to_numpy())
            for band in np.unique(TIME_BANDS):
                hours = np.flatnonzero(TIME_BANDS == band)
                keys = np.ix_(ids, hours)
                mean_gram = gram[keys].mean(axis=(0, 1))
                mean_cross = cross[keys].mean(axis=(0, 1))
                coefficient = np.linalg.solve(mean_gram + np.eye(len(FEATURES)) * 4, mean_cross).clip(-.2, .2)
                output[weekend][keys] = coefficient
                support[weekend][keys] = np.diag(mean_gram)
    return output, support


def forecast(history, future, origin, inputs):
    if history.date.max() != origin or future.date.min() <= origin or set(future) != {*KEY, "local_stack"}:
        raise ValueError("Use raw observations only through the origin and unlabelled archived forecasts.")
    meta, school, weather, ordinary, events = inputs
    mature = meta.first_date.eq(meta.first_date.min())
    station = meta.loc[mature]
    ids = station.index
    past, y, known = panel(history, "observed", ids, ordinary, events)
    dates, base, regular = panel(future, "local_stack", ids, ordinary, events)
    past_weather = weather_panel(past, station.weather_zone, weather)
    target_weather = weather_panel(dates, station.weather_zone, weather)
    earlier = pairs(past, y, known, past_weather, school)
    projected = pairs(dates, base, regular, target_weather, school)
    center = dates[len(dates) // 2]
    observed_beta, observed_support = response(earlier, station, center, origin=origin)
    forecast_beta, forecast_support = response(projected, station, center)
    support = np.minimum(observed_support, forecast_support)
    beta = (observed_beta - forecast_beta) * support / (support + 6)
    weekend = (dates.dayofweek >= 5).astype(int)
    raw = np.einsum("dshk,dshk->dsh", target_weather, beta[weekend], optimize=True)
    for day in range(7):
        same = dates.dayofweek == day
        eligible = regular[same]
        count = eligible.sum(axis=0)[:, None]
        amount = np.where(eligible[..., None], raw[same], 0.)
        mean = np.divide(amount.sum(axis=0), count, out=np.zeros_like(amount[0]), where=count > 0)
        raw[same] = np.where(eligible[..., None], raw[same] - mean, 0.)
    raw = raw.clip(-.18, .18)
    grid = pd.MultiIndex.from_product([dates, ids, HOURS], names=KEY)
    return pd.Series(raw.ravel(), index=grid).reindex(pd.MultiIndex.from_frame(future[KEY]), fill_value=0.).to_numpy()


def score(actual, reference, candidate, dates):
    first = np.square(actual - reference.clip(min=0))
    second = np.square(actual - candidate.clip(min=0))
    days = pd.DataFrame({"a": first, "b": second, "date": dates}).groupby("date")[["a", "b"]].mean().to_numpy()
    starts = np.random.default_rng(116).integers(len(days), size=(800, (len(days) + 6) // 7))
    positions = ((starts[..., None] + np.arange(7)) % len(days)).reshape(800, -1)[:, :len(days)]
    samples = np.sqrt(days[positions].mean(axis=1))
    interval = np.quantile(samples[:, 0] - samples[:, 1], [.025, .975])
    return {"reference": float(np.sqrt(first.mean())), "candidate": float(np.sqrt(second.mean())),
            "week_block_gain_interval": interval.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team", type=Path)
    parser.add_argument("earlier", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--recent", type=Path)
    args = parser.parse_args()
    work = args.work_dir
    inputs = context(work, args.team)
    station = inputs[0]
    raw = pd.read_csv(args.team / "data/train.csv", parse_dates=["date"], usecols=[*KEY, "boardings"])
    raw["observed"] = np.log1p(raw.pop("boardings"))
    true = raw.set_index(KEY).observed
    cols = [*KEY, "local_stack", "y_log"]
    origins = {name: pd.Timestamp(window[0]) for name, window in WINDOWS.items()}
    current = pd.read_csv(work / "sincon_local_validation.csv.gz", parse_dates=["date"], usecols=[*cols, "fold", "team"])
    summer = pd.read_csv(work / "sincon_summer_replay_validation.csv.gz", parse_dates=["date"], usecols=[*cols, "team"])
    summer_origin = pd.Timestamp("2025-06-30")
    archives = {name: (current.loc[current.fold.eq(name)].copy(), origin) for name, origin in origins.items()}
    archives["summer"] = (summer, summer_origin)
    for date in ("2025-07-31", "2025-08-31", "2025-09-30"):
        frame = pd.read_csv(args.earlier / f"origin_{date}.csv.gz", parse_dates=["date"], usecols=cols)
        archives[date] = (frame, pd.Timestamp(date))
    if args.recent:
        for path in sorted(args.recent.glob("origin_*.csv.gz")):
            date = path.name.removeprefix("origin_").removesuffix(".csv.gz")
            archives[date] = (pd.read_csv(path, parse_dates=["date"], usecols=cols), pd.Timestamp(date))
    calibration = pd.concat([current.assign(origin=current.fold.map(origins)), summer.assign(origin=summer_origin)], ignore_index=True)
    calibration["station_age"] = (calibration.origin - calibration.station_id.map(station.first_date)).dt.days

    report = {}
    for name, (rows, origin) in archives.items():
        rows = rows.reset_index(drop=True)
        index = pd.MultiIndex.from_frame(rows[KEY])
        if not index.is_unique or not np.allclose(rows.y_log, true.reindex(index), atol=1e-8, rtol=0):
            raise ValueError(f"Missing or inconsistent observed training labels in {name}.")
        delta = forecast(raw.loc[raw.date.le(origin)], rows[[*KEY, "local_stack"]], origin, inputs)
        if not np.isfinite(delta).all():
            raise ValueError(f"Non-finite weather deltas in {name}.")
        mature = rows.station_id.map(station.first_date).eq(station.first_date.min()).to_numpy()
        actual, base = rows.y_log.to_numpy(), rows.local_stack.to_numpy()
        primary = base + .45 * delta
        late = rows.hour.ge(21).to_numpy()
        late_only = base + .45 * np.where(late, delta, 0.)
        offsets, count = earlier_team_errors(calibration, origin)
        shift = rows.station_id.map(offsets).fillna(0).to_numpy()
        report[name] = {"origin": str(origin.date()), "rows": len(rows), "changed": int((np.abs(delta) > 1e-12).sum()),
                        "all": score(actual, base, primary, rows.date),
                        "mature_68": score(actual[mature], base[mature], primary[mature], rows.date[mature]),
                        "late": score(actual[late], base[late], primary[late], rows.date[late]),
                        "late_only_exploratory": score(actual, base, late_only, rows.date),
                        "late_only_team_calibration": score(actual, base + shift, late_only + shift, rows.date),
                        "prior_calibration_days": count}
        print(json.dumps({name: report[name]}, allow_nan=False), flush=True)
    (work / "sincon_weather_sensitivity_results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
