"""Model baseball-day ridership by aligning demand to the actual game finish."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import sincon_calendar_kernel as kernel
import sincon_event_effects as events
import sincon_external_covariates as external
import sincon_forecast as forecast
from sincon_lightgbm_experiments import HOLIDAY_FLAGS, attach_calendar, holiday_snapshot


GAME_FEATURES = ["hour", "weekday", "holiday_type", "start", "from_start", "exhibition", "postseason"]


def clock_hour(clock):
    time = pd.to_datetime(clock, format="%H:%M", errors="raise")
    return time.dt.hour + time.dt.minute / 60


def read_scoreboard(path, game_days):
    scores = pd.read_csv(path, usecols=["date", "game_id", "start_kst", "end_kst", "attendance"], parse_dates=["date"])
    if scores["game_id"].duplicated().any() or not scores["game_id"].str[:8].eq(scores["date"].dt.strftime("%Y%m%d")).all():
        raise ValueError("Each official game must have its own ID and matching date.")
    expected = game_days.loc[game_days["games_played"].gt(0), "games_played"].astype(int)
    if not scores.groupby("date").size().equals(expected.rename(None)):
        raise ValueError("Official scores must match the number of played games by date.")
    start = clock_hour(scores["start_kst"])
    end = clock_hour(scores["end_kst"])
    end = end.mask(end.lt(start), end + 24)
    if not np.isfinite(np.column_stack([start, end, scores["attendance"]])).all() or (end.le(start) | scores["attendance"].le(0)).any():
        raise ValueError("A played game needs a positive duration and official attendance.")
    return scores.assign(start=start, end=end).set_index("date").loc[lambda s: ~s.index.duplicated(keep=False)]


def game_inputs(frame, scores, game_days, weather):
    dated = frame.join(scores[["start", "end", "attendance"]], on="date", validate="many_to_one")
    dated = dated.join(game_days[["exhibition", "postseason"]], on="date", validate="many_to_one")
    dated = dated.join(weather, on=["date", "hour"], validate="many_to_one")
    return dated.assign(from_start=dated["hour"] + 0.5 - dated["start"],
                        from_end=dated["hour"] + 0.5 - dated["end"],
                        log_crowd=np.log(dated["attendance"]))


def rbf(left, right, column, scale):
    gap = (left[column].to_numpy()[:, None] - right[column].to_numpy()[None, :]) / scale
    return np.exp(-0.5 * np.square(gap))


def game_kernels(left, right):
    wall_hour = rbf(left, right, "hour", 1.25)
    start_relative = rbf(left, right, "from_start", 0.85)
    finish_relative = rbf(left, right, "from_end", 0.55)
    close_attendance = rbf(left, right, "log_crowd", 0.48)
    same_type = left["exhibition"].to_numpy()[:, None] == right["exhibition"].to_numpy()[None, :]
    similar_calendar = kernel.calendar_kernel(left, right)
    base = similar_calendar * (0.2 + wall_hour) + (1.4 + 1.0 * same_type) * start_relative
    finish = (2.5 + 1.25 * same_type) * finish_relative
    attendance = close_attendance * (0.35 * wall_hour + 0.9 * finish_relative)
    return base, finish, attendance


def exit_kernel(history, future):
    gram, finish_gram, attendance_gram = game_kernels(history, history)
    cross, finish_cross, attendance_cross = game_kernels(future, history)
    actual = np.log1p(history["boardings"].to_numpy(dtype="float64"))
    return {
        "start_kernel": kernel.kernel_predict(gram, cross, actual, penalty=2.0),
        "end_kernel": kernel.kernel_predict(gram + finish_gram, cross + finish_cross, actual, penalty=2.0),
        "end_crowd_kernel": kernel.kernel_predict(gram + finish_gram + attendance_gram,
                                                 cross + finish_cross + attendance_cross, actual, penalty=2.0),
    }


def exit_tree(history, future, threads):
    import lightgbm as lgb

    features = GAME_FEATURES + ["end", "from_end", "log_crowd", *external.RAIN]
    target = np.log1p(history["boardings"].to_numpy(dtype="float64"))
    model = lgb.train({"objective": "regression", "learning_rate": 0.045, "num_leaves": 15,
                       "min_data_in_leaf": 12, "lambda_l2": 6., "deterministic": True,
                       "force_col_wise": True, "num_threads": threads, "verbosity": -1, "seed": 42},
                      lgb.Dataset(history[features], label=target), num_boost_round=330)
    return model.predict(future[features], num_threads=threads)


def at_origin(history, future, calendar, weather, game_days, scores, concert_days, cutoff, threads):
    baseline, _ = external.predict_at(history, future, calendar, weather, game_days, cutoff, threads)
    snapshot = holiday_snapshot(calendar, cutoff)
    historical = attach_calendar(history, snapshot, cutoff)
    reference = events.event_forecasts(historical, future, concert_days, game_days, baseline["all_external"])[0]["event_venue_and_series"]
    stadium_events = concert_days.loc[concert_days["station_id"].eq(external.STADIUM), "date"]
    route = future["station_id"].eq(external.STADIUM) & future["date"].isin(scores.index) & ~future["date"].isin(stadium_events)
    result = {"reference": reference}
    names = ["start_kernel", "end_kernel", "end_crowd_kernel", "end_crowd_tree", "exit_blend"]
    if not route.any():
        return result | {name: reference for name in names}, route.to_numpy()
    trained = historical.loc[historical["station_id"].eq(external.STADIUM) & historical["date"].isin(scores.index)
                             & ~historical["date"].isin(stadium_events)]
    target = attach_calendar(future.loc[route], snapshot, cutoff)
    trained = game_inputs(trained, scores, game_days, weather)
    target = game_inputs(target, scores, game_days, weather)
    trained = trained.merge(kernel.dates_and_regimes(trained)[["date", "regime"]], on="date", validate="many_to_one")
    target = target.merge(kernel.dates_and_regimes(target)[["date", "regime"]], on="date", validate="many_to_one")
    candidates = exit_kernel(trained, target)
    candidates["end_crowd_tree"] = exit_tree(trained, target, threads)
    candidates["exit_blend"] = (0.65 * candidates["end_crowd_kernel"] + 0.25 * candidates["end_crowd_tree"]
                                + 0.10 * np.log1p(reference[route]))
    for name, log_values in candidates.items():
        values = reference.copy()
        values[route] = np.expm1(np.maximum(log_values, 0))
        result[name] = values
    return result, route.to_numpy()


def run(data_zip, calendar_csv, external_zip, scores_csv, threads, submission):
    train, test = forecast.read_inputs(data_zip)
    dates = pd.date_range(train["date"].min(), test["date"].max(), name="date")
    calendar = pd.read_csv(calendar_csv, usecols=["date", *HOLIDAY_FLAGS], parse_dates=["date"]).set_index("date").sort_index()
    weather, game_days = external.read_external(external_zip, dates)
    scores = read_scoreboard(scores_csv, game_days)
    concert_days = events.read_events(external_zip)
    reports = {}
    errors = {}
    for name, cutoff in forecast.BACKTESTS.items():
        started = perf_counter()
        history, validation = forecast.split_backtest(train, cutoff)
        future = validation.drop(columns="boardings")
        predictions, route = at_origin(history, future, calendar, weather, game_days, scores, concert_days, cutoff, threads)
        reports[name] = {"rows": len(validation), "baseball_rows": int(route.sum()),
                         "rmsle": {model: forecast.rmsle(validation["boardings"], values) for model, values in predictions.items()},
                         "baseball_rmsle": {model: forecast.rmsle(validation.loc[route, "boardings"], values[route])
                                            for model, values in predictions.items()},
                         "seconds": round(perf_counter() - started, 2)}
        errors[name] = validation[["station_id", "hour"]].assign(**{
            model: np.square(np.log1p(validation["boardings"].to_numpy(dtype="float64")) - np.log1p(values))
            for model, values in predictions.items()
        })
        print(json.dumps({name: reports[name]}, allow_nan=False), flush=True)
    pooled = {model: float(np.sqrt(np.average([report["rmsle"][model]**2 for report in reports.values()],
                                             weights=[report["rows"] for report in reports.values()]))) for model in predictions}
    best = min(pooled, key=pooled.get)
    errors = pd.concat(errors.values())
    error_shares = {column: (errors.groupby(column)[best].sum() / errors[best].sum()).nlargest(10).to_dict()
                    for column in ["station_id", "hour"]}
    final = None
    if submission:
        predictions, route = at_origin(train, test, calendar, weather, game_days, scores, concert_days, forecast.CUTOFF, threads)
        forecast.write_submission(forecast.make_submission(test, predictions[best]), submission)
        final = {"model": best, "rows": len(test), "baseball_rows": int(route.sum())}
    return {"folds": reports, "pooled_rmsle": pooled, "best": best, "error_shares": error_shares, "final": final}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_zip", type=Path)
    parser.add_argument("calendar_csv", type=Path)
    parser.add_argument("external_zip", type=Path)
    parser.add_argument("scores_csv", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    result = run(args.data_zip, args.calendar_csv, args.external_zip, args.scores_csv, args.threads, args.submission)
    if args.results:
        args.results.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["pooled_rmsle", "best", "error_shares", "final"]}, allow_nan=False), flush=True)
