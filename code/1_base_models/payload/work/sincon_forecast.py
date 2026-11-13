"""Compare two fixed-origin SINCON forecasts and write a Kaggle submission."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp("2026-06-30")
HORIZON = 123
BACKTESTS = {
    "same_season": pd.Timestamp("2025-06-30"),
    "recent": pd.Timestamp("2026-02-27"),
}
KEY = ["date", "station_id", "hour"]

# Korean public holidays, including substitutes and election days. Sources:
# KASA's calendar standards and later government holiday designations.
# https://www.kasa.go.kr/prog/bbsArticle/BBSMSTR_000000000010/view.do?nttId=B000000000633Wg5pO7
# https://www.kasa.go.kr/prog/plcyBrf/brief/kor/sub01_01_04/view.do?plcyBrfNo=202
# https://www.korea.kr/news/policyNewsView.do?newsId=148938559
# https://www.korea.kr/news/policyNewsView.do?newsId=148941443
# https://law.go.kr/LSW/lsRvsRsnListP.do?chrClsCd=010102&lsId=002404
HOLIDAYS = pd.to_datetime("""
    2025-01-01 2025-01-27 2025-01-28 2025-01-29 2025-01-30
    2025-03-01 2025-03-03 2025-05-05 2025-05-06 2025-06-03 2025-06-06
    2025-08-15 2025-10-03 2025-10-05 2025-10-06 2025-10-07
    2025-10-08 2025-10-09 2025-12-25
    2026-01-01 2026-02-16 2026-02-17 2026-02-18 2026-03-01 2026-03-02
    2026-05-01 2026-05-05 2026-05-24 2026-05-25 2026-06-03 2026-06-06
    2026-07-17 2026-08-15 2026-08-17 2026-09-24 2026-09-25 2026-09-26
    2026-10-03 2026-10-05 2026-10-09
""".split())

LUNAR_HOLIDAYS = pd.to_datetime("""
    2025-01-28 2025-01-29 2025-01-30
    2025-10-05 2025-10-06 2025-10-07 2025-10-08
    2026-02-16 2026-02-17 2026-02-18
    2026-09-24 2026-09-25 2026-09-26
""".split())

CALENDAR_PUBLICATION = {2025: "2024-06-20", 2026: "2025-06-30"}
HOLIDAY_CONFIRMATION = {
    "2025-01-27": "2025-01-14",  # Temporary Lunar New Year holiday.
    "2025-06-03": "2025-04-08",  # Presidential election.
    "2026-05-01": "2026-04-09",  # Labour Day added to the public holiday law.
    "2026-07-17": "2026-02-10",  # Constitution Day reinstated.
}

CATEGORICAL = ["station_id", "canonical_line", "weekday", "month", "holiday_type"]
FEATURES = CATEGORICAL + ["hour", "holiday_eve", "after_holiday"]
LIGHTGBM_ROUNDS = 450


def holidays_at(cutoff):
    return pd.DatetimeIndex([
        day for day in HOLIDAYS
        if pd.Timestamp(HOLIDAY_CONFIRMATION.get(
            day.strftime("%Y-%m-%d"), CALENDAR_PUBLICATION[day.year]
        )) <= cutoff
    ])


def add_calendar(frame, cutoff=CUTOFF):
    date = frame["date"]
    weekday = date.dt.dayofweek
    known_holidays = holidays_at(cutoff)
    holiday = date.isin(known_holidays)
    return frame.assign(
        weekday=weekday,
        month=date.dt.month,
        holiday_type=holiday.astype("int8") + (holiday & date.isin(LUNAR_HOLIDAYS)).astype("int8"),
        holiday_eve=(date + pd.Timedelta(days=1)).isin(known_holidays).astype("int8"),
        after_holiday=(date - pd.Timedelta(days=1)).isin(known_holidays).astype("int8"),
        day_group=weekday.mask(holiday, 7),
        non_workday=(weekday >= 5) | holiday,
    )


def read_inputs(path):
    with ZipFile(path) as archive:
        train = pd.read_csv(
            archive.open("train.csv"), usecols=KEY + ["boardings"], parse_dates=["date"]
        )
        test = pd.read_csv(archive.open("test.csv"), parse_dates=["date"])
        stations = pd.read_csv(
            archive.open("metaData.csv"), usecols=["station_id", "canonical_line"]
        )

    expected_dates = pd.date_range(CUTOFF + pd.Timedelta(days=1), periods=HORIZON)
    if (
        test["example_id"].isna().any()
        or not test["example_id"].is_unique
        or test.duplicated(KEY).any()
        or set(test["date"]) != set(expected_dates)
        or set(test["station_id"]) != set(stations["station_id"])
        or len(stations) != 71
        or set(test["hour"]) != set(range(6, 24))
        or len(test) != HORIZON * 71 * 18
    ):
        raise ValueError("test.csv must contain a unique ID for every target date, station and hour.")
    if (
        train["date"].max() != CUTOFF
        or train.duplicated(KEY).any()
        or not np.isfinite(train["boardings"]).all()
        or (train["boardings"] < 0).any()
    ):
        raise ValueError("Invalid training dates, keys or boarding counts.")

    return tuple(
        add_calendar(frame.merge(stations, on="station_id", how="left", validate="many_to_one"))
        for frame in (train, test)
    )


def split_backtest(train, cutoff):
    horizon_end = cutoff + pd.Timedelta(days=HORIZON)
    history = train.loc[train["date"] <= cutoff]
    validation = train.loc[train["date"].gt(cutoff) & train["date"].le(horizon_end)]
    if history.empty or validation["date"].nunique() != HORIZON:
        raise ValueError(f"Cannot run a complete {HORIZON}-day backtest at {cutoff.date()}.")
    return history, validation


def seasonal_predict(history, future):
    log_boardings = np.log1p(history["boardings"].astype("float64"))
    history = history.assign(log_boardings=log_boardings)
    estimates = pd.Series(np.nan, index=future.index)
    for keys in (
        ["station_id", "hour", "day_group"],
        ["station_id", "hour", "non_workday"],
        ["station_id", "hour"],
        ["canonical_line", "hour", "day_group"],
    ):
        missing = estimates.isna()
        if not missing.any():
            break
        averages = history.groupby(keys)["log_boardings"].mean()
        keys_to_predict = pd.MultiIndex.from_frame(future.loc[missing, keys])
        estimates.loc[missing] = averages.reindex(keys_to_predict).to_numpy()
    return np.expm1(estimates.fillna(log_boardings.mean()).to_numpy())


def lightgbm_inputs(frame, features):
    category_types = {name: "category" for name in CATEGORICAL if name in features}
    return frame[features].astype(category_types)


def fit_lightgbm(
    history, threads, *, features=FEATURES, learning_rate=0.05, lambda_l2=1.0,
    rounds=LIGHTGBM_ROUNDS, stopping_validation=None,
):
    import lightgbm as lgb

    training_set = lgb.Dataset(
        lightgbm_inputs(history, features),
        label=np.log1p(history["boardings"].to_numpy(dtype="float64")),
    )
    params = {
        "objective": "regression",
        "device_type": "cpu",
        "learning_rate": learning_rate,
        "num_leaves": 63,
        "min_data_in_leaf": 80,
        "lambda_l2": lambda_l2,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": threads,
        "seed": 42,
        "verbosity": -1,
    }
    if stopping_validation is None:
        return lgb.train(params, training_set, num_boost_round=rounds)

    validation_set = training_set.create_valid(
        lightgbm_inputs(stopping_validation, features),
        label=np.log1p(stopping_validation["boardings"].to_numpy(dtype="float64")),
    )
    params["metric"] = "None"

    def validation_rmsle(log_predictions, dataset):
        log_error = dataset.get_label() - np.maximum(log_predictions, 0)
        return "rmsle", float(np.sqrt(np.mean(np.square(log_error)))), False

    return lgb.train(
        params, training_set, num_boost_round=rounds, valid_sets=[validation_set],
        feval=validation_rmsle,
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )


def lightgbm_predict(history, future, threads, *, features=FEATURES, **training_options):
    model = fit_lightgbm(history, threads, features=features, **training_options)
    log_predictions = model.predict(lightgbm_inputs(future, features), num_threads=threads)
    return np.expm1(np.maximum(log_predictions, 0))


def rmsle(actual, predicted):
    log_error = np.log1p(np.asarray(actual, dtype="float64")) - np.log1p(predicted)
    return float(np.sqrt(np.mean(np.square(log_error))))


def make_submission(test, predictions):
    predictions = np.asarray(predictions, dtype="float64")
    if (
        predictions.shape != (len(test),)
        or not np.isfinite(predictions).all()
        or (predictions < 0).any()
    ):
        raise ValueError("Each test ID must have one finite, nonnegative prediction.")
    return pd.DataFrame({"example_id": test["example_id"], "boardings": predictions})


def write_submission(submission, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            submission.to_csv(handle, index=False)
            handle.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


def run(data_path, output_path, threads):
    train, test = read_inputs(data_path)
    models = {
        "seasonal_average": seasonal_predict,
        "cpu_lightgbm": lambda history, future: lightgbm_predict(history, future, threads),
    }
    scores = {name: [] for name in models}
    counts = []
    for fold, cutoff in BACKTESTS.items():
        history, validation = split_backtest(train, cutoff)
        history = add_calendar(history, cutoff)
        validation = add_calendar(validation, cutoff)
        future = validation.drop(columns="boardings")
        counts.append(len(validation))
        for name, predict in models.items():
            started = perf_counter()
            predictions = predict(history, future)
            score = rmsle(validation["boardings"], predictions)
            scores[name].append(score)
            print(json.dumps({
                "fold": fold,
                "model": name,
                "train_end": str(cutoff.date()),
                "validation_start": str(validation["date"].min().date()),
                "validation_end": str(validation["date"].max().date()),
                "train_rows": len(history),
                "validation_rows": len(validation),
                "rmsle": score,
                "seconds": round(perf_counter() - started, 2),
            }), flush=True)

    pooled = {
        name: float(np.sqrt(np.average(np.square(fold_scores), weights=counts)))
        for name, fold_scores in scores.items()
    }
    winner = min(pooled, key=pooled.__getitem__)
    print(json.dumps({"pooled_rmsle": pooled, "selected_model": winner}), flush=True)
    started = perf_counter()
    predictions = models[winner](train, test.drop(columns="example_id"))
    submission = make_submission(test, predictions)
    write_submission(submission, output_path)
    print(json.dumps({
        "submission": str(output_path.resolve()),
        "rows": len(submission),
        "refit_and_export_seconds": round(perf_counter() - started, 2),
    }), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_zip", type=Path, help="SINCON archive containing train/test/metadata CSVs.")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    run(args.data_zip, args.output, args.threads)


if __name__ == "__main__":
    main()
