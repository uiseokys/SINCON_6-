"""Backtest common demand shocks using bus, walking and car arrival estimates."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sincon_mobility_data import MODES
from sincon_station_state import EVENT_STATIONS, REGULAR, WINDOWS, read_team


BANDS = ["morning", "midday", "evening", "late"]


def band(hours):
    return pd.cut(hours, bins=[5, 9, 16, 20, 23], labels=BANDS)


def load_inputs(directory, external_zip, calendar):
    paths = sorted(directory.glob("*.csv.gz"))
    if not paths:
        raise ValueError("No mobility month summaries were found.")
    mobility = pd.concat((pd.read_csv(path, parse_dates=["date"], dtype={"district": str}) for path in paths), ignore_index=True)
    if set(mobility["mode"]) != set(MODES.values()) or mobility.duplicated(["date", "hour", "district", "mode"]).any():
        raise ValueError("Expected unique aggregate records containing the four approved modes.")
    mobility = mobility.assign(band=band(mobility["hour"]), city=np.where(mobility["district"].str.startswith("28"), "incheon", "bucheon"))
    mobility = mobility.groupby(["date", "band", "city", "mode"], observed=True)["arrivals"].sum().unstack(["city", "mode"])
    grid = pd.MultiIndex.from_product([calendar.index, BANDS], names=["date", "band"])
    mobility = mobility.reindex(grid)
    if mobility.isna().any().any() or mobility.lt(0).any().any():
        raise ValueError("Mobility must cover both cities, all dates and every time band.")
    with ZipFile(external_zip) as archive:
        weather = pd.read_csv(archive.open("external/weather_hourly.csv"), parse_dates=["date"],
                              usecols=["date", "hour", "precip", "temp_feel", "snowfall"])
    weather["band"] = band(weather["hour"])
    weather = weather.groupby(["date", "band"], observed=True)[["precip", "temp_feel", "snowfall"]].mean().reindex(grid)
    if weather.isna().any().any():
        raise ValueError("Weather must cover every date and time band.")
    weather[["precip", "snowfall"]] = np.log1p(weather[["precip", "snowfall"]])
    return np.log1p(mobility), weather


def local_anomalies(values, dates, ordinary):
    days = dates.to_numpy(dtype="datetime64[D]").astype("int64")
    gap = np.abs(days[:, None] - days[None, :])
    weekday = dates.dayofweek.to_numpy()
    peers = (weekday[:, None] == weekday[None, :]) & (gap > 0) & (gap <= 42) & ordinary[None, :]
    eligible = peers.sum(axis=1) >= 3
    average = np.zeros_like(values)
    for row in np.flatnonzero(eligible):
        average[row] = np.median(values[peers[row]], axis=0)
    result = values - average
    result[~eligible] = 0
    return result, eligible


def ridge_predict(x, y, target):
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 0.02)
    design = (x - center) / scale
    beta = np.linalg.solve(design.T @ design + 18 * np.eye(design.shape[1]), design.T @ (y - y.mean()))
    return (target - center) / scale @ beta + y.mean()


def effect_fit(historical, target, mobility, weather, calendar, by_band):
    cut = historical["date"].max()
    target_dates = pd.DatetimeIndex(target["date"].unique()).sort_values()
    if cut >= target_dates.min() or {"y_log", "boardings"} & set(target):
        raise ValueError("Demand observations must end before the unlabeled forecast period.")
    historic_dates = pd.date_range(historical["date"].min(), cut)
    ordinary = calendar[REGULAR].eq(0).all(axis=1)
    historical = historical.assign(band=band(historical["hour"]), log=np.log1p(historical["boardings"]))
    result = pd.DataFrame(index=pd.MultiIndex.from_product([target_dates, BANDS], names=["date", "band"]), columns=["shift"], dtype=float)
    for part in BANDS if by_band else [None]:
        if part is None:
            row = historical
            modes = mobility.groupby(level="date").mean()
            conditions = weather.groupby(level="date").mean()
        else:
            row = historical.loc[historical["band"].eq(part)]
            modes = mobility.xs(part, level="band")
            conditions = weather.xs(part, level="band")
        levels = row.groupby("date")["log"].mean().reindex(historic_dates)
        if levels.isna().any():
            raise ValueError("Historical ridership must have a complete daily panel for the reference stations.")
        y, y_ok = local_anomalies(levels.to_numpy()[:, None], historic_dates, ordinary.loc[historic_dates].to_numpy())
        dates = pd.DatetimeIndex(modes.index)
        day_ordinary = ordinary.loc[dates].to_numpy()
        x, x_ok = local_anomalies(np.column_stack([conditions, modes]), dates, day_ordinary)
        x = pd.DataFrame(x, index=dates)
        eligible = y_ok & ordinary.loc[historic_dates].to_numpy() & x_ok[dates.get_indexer(historic_dates)]
        design = x.loc[historic_dates].to_numpy()[eligible]
        forward = x.loc[target_dates].to_numpy()
        n_weather = conditions.shape[1]
        controls = ridge_predict(design[:, :n_weather], y[eligible, 0], forward[:, :n_weather])
        full = ridge_predict(design, y[eligible, 0], forward)
        shift = np.clip(full - controls, -0.12, 0.12) * ordinary.loc[target_dates].to_numpy()
        selected = BANDS if part is None else [part]
        for slot in selected:
            result.loc[pd.IndexSlice[:, slot], "shift"] = shift
    keys = pd.MultiIndex.from_arrays([target["date"], band(target["hour"])], names=["date", "band"])
    values = result["shift"].reindex(keys).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Every forecast row needs a finite mobility adjustment.")
    return values


def run(team_dir, external_zip, directory):
    train, meta, calendar, predictions = read_team(team_dir, external_zip)
    mobility, weather = load_inputs(directory, external_zip, calendar)
    mature = meta.index[meta["first_date"].eq(meta["first_date"].min()) & ~meta.index.isin(EVENT_STATIONS)]
    stable = train.loc[train["station_id"].isin(mature)]
    report = {}
    for name, (cutoff, _) in WINDOWS.items():
        earlier = stable.loc[stable["date"].le(cutoff)]
        fold = predictions.loc[predictions["fold"].eq(name)]
        target = fold[["date", "station_id", "hour", "ens_log"]]
        base = fold["ens_log"].to_numpy()
        actual = fold["y_log"].to_numpy()
        forecast_logs = {"ensemble": base}
        for by_band in (False, True):
            adjustment = effect_fit(earlier, target, mobility, weather, calendar, by_band)
            forecast_logs["mobility_by_time_band" if by_band else "mobility_daily"] = base + adjustment
        daily_error = fold[["date"]].assign(**{key: actual - np.maximum(value, 0) for key, value in forecast_logs.items()})
        report[name] = {"rmsle": {key: float(np.sqrt(np.mean((actual - np.maximum(value, 0)) ** 2)))
                                  for key, value in forecast_logs.items()},
                        "daily_error_rmse": daily_error.groupby("date").mean().pow(2).mean().pow(0.5).to_dict()}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team_dir", type=Path)
    parser.add_argument("external_zip", type=Path)
    parser.add_argument("mobility_directory", type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    output = json.dumps(run(args.team_dir, args.external_zip, args.mobility_directory), indent=2, allow_nan=False) + "\n"
    if args.results:
        args.results.write_text(output, encoding="utf-8")
    print(output, end="")
