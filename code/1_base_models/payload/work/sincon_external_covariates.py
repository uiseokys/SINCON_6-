"""Backtest actual weather and baseball covariates without future ridership."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd

import sincon_calendar_kernel as kernel
import sincon_forecast as forecast
from sincon_hierarchical_experiments import MIN_HISTORY_DAYS
from sincon_intraday_ensemble import EXTRA_FOLD
from sincon_lightgbm_experiments import HOLIDAY_FLAGS, attach_calendar, holiday_snapshot


HOURS = range(6, 24)
STADIUM = 3127
RAIN = ["precip", "precip_prev3h", "day_precip_6_23", "snowfall"]
TEMPERATURE = ["temp_feel", "day_temp_max"]
GAME = ["games_played", "cancelled", "postseason", "exhibition", "start_hour"]


def summarize_games(records, dates):
    actual = records.loc[records["source"].str.startswith("kbo_result_")].copy()
    played = actual.loc[actual["status"].eq("played")].copy()
    time = pd.to_datetime(played["start_time_kst"], format="%H:%M", errors="raise")
    if time.isna().any():
        raise ValueError("Played games must have a start time.")
    played = played.assign(start_hour=time.dt.hour + time.dt.minute / 60,
                           postseason=played["game_type"].eq("postseason"),
                           exhibition=played["game_type"].eq("exhibition"))
    summary = played.groupby("game_date").agg(
        games_played=("start_hour", "size"), start_hour=("start_hour", "min"),
        postseason=("postseason", "max"), exhibition=("exhibition", "max"),
    )
    cancelled = actual.loc[actual["status"].eq("cancelled"), "game_date"]
    summary = summary.reindex(dates)
    summary["cancelled"] = dates.isin(cancelled) & summary["games_played"].isna()
    flags = GAME[:-1]
    summary[flags] = summary[flags].astype("float64").fillna(0)
    return summary[GAME]


def read_external(path, dates):
    with ZipFile(path) as archive:
        weather = pd.read_csv(archive.open("external/weather_hourly.csv"),
                              usecols=["date", "hour", *RAIN, *TEMPERATURE], parse_dates=["date"])
        games = pd.read_csv(archive.open("external/munhak_games.csv"),
                            usecols=["game_date", "start_time_kst", "game_type", "status", "source"],
                            parse_dates=["game_date"])
    weather = weather.set_index(["date", "hour"])[RAIN + TEMPERATURE]
    expected = pd.MultiIndex.from_product([dates, HOURS], names=["date", "hour"])
    weather = weather.reindex(expected)
    if not np.isfinite(weather.to_numpy()).all():
        raise ValueError("Weather must be available for every modeled date and hour.")
    for year, part in games.loc[games["source"].str.startswith("kbo_result_")].groupby("source"):
        if set(part["status"]) - {"played", "cancelled"}:
            raise ValueError(f"Unexpected game status in {year}.")
    return weather, summarize_games(games, dates)


def distance(left, right, scales):
    diff = (left[:, None, :] - right[None, :, :]) / scales
    return np.square(diff).sum(axis=2)


def weather_kernels(left, right, left_days, right_days):
    same_offday = (left_days["regime"].to_numpy()[:, None] == 0) == (right_days["regime"].to_numpy()[None, :] == 0)
    regime_weight = 0.5 + 0.5 * same_offday
    rain = np.log1p(left[:, :len(RAIN)])
    rain_right = np.log1p(right[:, :len(RAIN)])
    rain_similarity = np.exp(-0.5 * distance(rain, rain_right, [0.85, 1.3, 2.4, 0.7]))
    temp_similarity = np.exp(-0.5 * distance(left[:, len(RAIN):], right[:, len(RAIN):], [9, 12]))
    return 0.55 * regime_weight * rain_similarity, 0.25 * regime_weight * temp_similarity


def weather_adjustments(history, future, weather):
    source_days = kernel.dates_and_regimes(history)
    target_days = kernel.dates_and_regimes(future)
    gram = kernel.calendar_kernel(source_days, source_days)
    cross = kernel.calendar_kernel(target_days, source_days)
    first_dates = history.groupby("station_id")["date"].min()
    counts = history.groupby("station_id")["date"].nunique()
    warm_dates = first_dates[counts.ge(MIN_HISTORY_DAYS)]
    changes = np.zeros((len(future), 2))

    for hour in HOURS:
        w_history = weather.xs(hour, level="hour").loc[source_days["date"]].to_numpy()
        w_future = weather.xs(hour, level="hour").loc[target_days["date"]].to_numpy()
        rain_gram, temp_gram = weather_kernels(w_history, w_history, source_days, source_days)
        rain_cross, temp_cross = weather_kernels(w_future, w_history, target_days, source_days)
        counts_at_hour = history.loc[history["hour"].eq(hour)].pivot(
            index="date", columns="station_id", values="boardings",
        )
        for first_date, stations in warm_dates.groupby(warm_dates).groups.items():
            start = int(source_days["date"].searchsorted(first_date))
            logs = np.log1p(counts_at_hour.loc[first_date:, stations].to_numpy(dtype="float64"))
            base_gram = gram[start:, start:]
            base_cross = cross[:, start:]
            rain = rain_gram[start:, start:]
            future_rain = rain_cross[:, start:]
            base = kernel.kernel_predict(base_gram, base_cross, logs, penalty=2.0)
            wet = kernel.kernel_predict(base_gram + rain, base_cross + future_rain, logs, penalty=2.0)
            all_weather = kernel.kernel_predict(base_gram + rain + temp_gram[start:, start:],
                                                base_cross + future_rain + temp_cross[:, start:], logs, penalty=2.0)
            target = future["hour"].eq(hour) & future["station_id"].isin(stations)
            target_index = pd.MultiIndex.from_frame(future.loc[target, ["date", "station_id"]])
            keys = pd.MultiIndex.from_product([target_days["date"], stations], names=["date", "station_id"])
            for col, delta in enumerate((wet - base, all_weather - base)):
                changes[target, col] = pd.Series(delta.ravel(), index=keys).reindex(target_index).to_numpy()
    return changes


def game_kernel(left, right):
    played = left["games_played"].to_numpy()[:, None] > 0
    played_right = right["games_played"].to_numpy()[None, :] > 0
    start = left["start_hour"].fillna(0).to_numpy()[:, None]
    start_right = right["start_hour"].fillna(0).to_numpy()[None, :]
    start_match = np.exp(-0.5 * np.square((start - start_right) / 1.0))
    cancelled = left["cancelled"].to_numpy()[:, None] * right["cancelled"].to_numpy()[None, :]
    exhibition = left["exhibition"].to_numpy()[:, None] * right["exhibition"].to_numpy()[None, :]
    return played * played_right * (0.8 + 3 * start_match) + 0.45 * cancelled + 1.8 * exhibition


def stadium_kernel(history, future, games):
    source_days = kernel.dates_and_regimes(history).join(games, on="date", validate="one_to_one")
    target_days = kernel.dates_and_regimes(future).join(games, on="date", validate="one_to_one")
    actual = history.pivot(index="date", columns="hour", values="boardings").loc[source_days["date"], HOURS]
    actual = np.log1p(actual.to_numpy(dtype="float64"))
    gram = kernel.calendar_kernel(source_days, source_days) + game_kernel(source_days, source_days)
    cross = kernel.calendar_kernel(target_days, source_days) + game_kernel(target_days, source_days)
    log_predictions = kernel.kernel_predict(gram, cross, actual, penalty=1.25)
    index = pd.MultiIndex.from_product([target_days["date"], HOURS], names=["date", "hour"])
    return pd.Series(log_predictions.ravel(), index=index).reindex(pd.MultiIndex.from_frame(future[["date", "hour"]])).to_numpy()


def stadium_tree(history, future, games, weather, threads):
    import lightgbm as lgb

    features = ["hour", "weekday", "holiday_type", "is_lunar_season", "is_bridge_day", "season_sin", "season_cos",
                *GAME, "hours_from_start", *RAIN]

    def inputs(frame):
        dated = frame.join(games, on="date", validate="many_to_one").join(weather, on=["date", "hour"], validate="many_to_one")
        angle = 2 * np.pi * dated["date"].dt.dayofyear / 365.2425
        return dated.assign(season_sin=np.sin(angle), season_cos=np.cos(angle),
                            hours_from_start=dated["hour"] - dated["start_hour"])[features]

    training = lgb.Dataset(inputs(history), label=np.log1p(history["boardings"].to_numpy(dtype="float64")))
    model = lgb.train({"objective": "regression", "learning_rate": 0.045, "num_leaves": 23,
                       "min_data_in_leaf": 20, "lambda_l2": 6., "deterministic": True,
                       "force_col_wise": True, "num_threads": threads, "verbosity": -1, "seed": 42},
                      training, num_boost_round=400)
    return model.predict(inputs(future), num_threads=threads)


def predict_at(history, future, calendar, weather, games, cutoff, threads):
    if history["date"].max() > cutoff or future["date"].min() <= cutoff or {"boardings", "alightings"} & set(future):
        raise ValueError("Expected pre-cutoff ridership and unlabeled future rows.")
    prior, calibration_end = kernel.predict_at(history, future, calendar, cutoff, threads)
    reference = np.log1p(prior["kernel_blend"])
    snapshot = holiday_snapshot(calendar, cutoff)
    source = attach_calendar(history, snapshot, cutoff)
    target = attach_calendar(future, snapshot, cutoff)
    rain, rain_temp = weather_adjustments(source, target, weather).T
    stadium = target["station_id"].eq(STADIUM).to_numpy()
    source_stadium = source.loc[source["station_id"].eq(STADIUM)]
    target_stadium = target.loc[stadium]
    events = stadium_kernel(source_stadium, target_stadium, games)
    tree = stadium_tree(source_stadium, target_stadium, games, weather, threads)
    blended_stadium = 0.70 * events + 0.30 * tree
    logs = {
        "calendar_reference": reference,
        "rain": reference + 0.75 * rain,
        "rain_and_temperature": reference + 0.75 * rain_temp,
        "baseball_kernel": reference.copy(),
        "baseball_and_weather_station_tree": reference.copy(),
        "rain_and_baseball_kernel": reference + 0.75 * rain,
        "all_external": reference + 0.75 * rain_temp,
    }
    logs["baseball_kernel"][stadium] = events
    logs["baseball_and_weather_station_tree"][stadium] = tree
    logs["rain_and_baseball_kernel"][stadium] = events + 0.75 * rain[stadium]
    logs["all_external"][stadium] = blended_stadium + 0.50 * rain_temp[stadium]
    return {name: np.expm1(np.maximum(values, 0)) for name, values in logs.items()}, calibration_end


def run(data_zip, calendar_csv, external_zip, threads, submission):
    train, test = forecast.read_inputs(data_zip)
    dates = pd.date_range(train["date"].min(), test["date"].max(), name="date")
    calendar = pd.read_csv(calendar_csv, usecols=["date", *HOLIDAY_FLAGS], parse_dates=["date"]).set_index("date").sort_index()
    if not calendar.index.equals(dates) or calendar.isna().any().any():
        raise ValueError("Calendar must cover every train and test date once.")
    weather, games = read_external(external_zip, dates)
    folds = {**forecast.BACKTESTS, "autumn_origin": pd.Timestamp("2025-10-31"), **EXTRA_FOLD}
    results = {}
    for fold, cutoff in folds.items():
        started = perf_counter()
        history, validation = forecast.split_backtest(train, cutoff)
        future = validation.drop(columns="boardings")
        predictions, calibration_end = predict_at(history, future, calendar, weather, games, cutoff, threads)
        stadium = validation["station_id"].eq(STADIUM).to_numpy()
        results[fold] = {
            "cutoff": str(cutoff.date()), "calibration_end": calibration_end, "rows": len(validation),
            "rmsle": {name: forecast.rmsle(validation["boardings"], values) for name, values in predictions.items()},
            "stadium_rmsle": {name: forecast.rmsle(validation.loc[stadium, "boardings"], values[stadium])
                              for name, values in predictions.items()},
            "seconds": round(perf_counter() - started, 2),
        }
        print(json.dumps({fold: results[fold]}, allow_nan=False), flush=True)
    pooled = {name: float(np.sqrt(np.average([results[fold]["rmsle"][name]**2 for fold in forecast.BACKTESTS],
                                             weights=[results[fold]["rows"] for fold in forecast.BACKTESTS])))
              for name in predictions}
    best = min(pooled, key=pooled.get)
    if submission:
        started = perf_counter()
        predictions, calibration_end = predict_at(train, test, calendar, weather, games, forecast.CUTOFF, threads)
        forecast.write_submission(forecast.make_submission(test, predictions[best]), submission)
        final = {"model": best, "calibration_end": calibration_end, "rows": len(test), "seconds": round(perf_counter() - started, 2)}
    else:
        final = None
    return {"folds": results, "pooled_existing_folds": pooled, "best": best, "final": final}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_zip", type=Path)
    parser.add_argument("calendar_csv", type=Path)
    parser.add_argument("external_zip", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    output = run(args.data_zip, args.calendar_csv, args.external_zip, args.threads, args.submission)
    if args.results:
        args.results.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"pooled_existing_folds": output["pooled_existing_folds"], "best": output["best"],
                      "final": output["final"]}, allow_nan=False), flush=True)
