"""Replay the team's XGBoost on a fully historical fixed-origin window."""

import argparse
import importlib.util
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


KEY = ["date", "station_id", "hour"]
ORIGIN = "2025-06-30"
END = "2025-10-31"


def prepare_inputs(team, origin, end, competition_test=False):
    source = team / "experiments/model_zoo.py"
    spec = importlib.util.spec_from_file_location("team_model_zoo", source)
    zoo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(zoo)
    official = pd.read_csv(team / "data/train.csv", usecols=[*KEY, "boardings"])
    metadata = pd.read_csv(team / "data/metaData.csv", usecols=["station_id", "canonical_line"]).rename(
        columns={"canonical_line": "line"})
    hist = official.loc[official.date.le(origin)]
    if competition_test:
        if (origin, end) != ("2026-06-30", "2026-10-31"):
            raise ValueError("The competition forecast uses the published June 30 origin and October 31 end.")
        holdout = pd.read_csv(team / "data/test.csv", usecols=KEY)
    else:
        holdout = official.loc[official.date.gt(origin) & official.date.le(end), KEY]
    dates = pd.date_range(pd.Timestamp(origin) + pd.Timedelta(days=1), end)
    if (not len(dates) or len(holdout) != len(dates) * 71 * 18 or holdout.duplicated(KEY).any()
            or set(holdout.date) != set(dates.strftime("%Y-%m-%d"))
            or hist.date.max() != origin or official.date.max() != "2026-06-30"):
        raise ValueError("Unexpected fixed-origin training or target keys.")
    calendar = zoo.build_calendar()
    fit = zoo.im.enrich(hist, calendar, metadata, True)
    forecast = zoo.im.enrich(holdout, calendar, metadata, True)
    if "boardings" in forecast or fit.date.max() >= forecast.date.min():
        raise ValueError("The XGBoost evaluation requires a strictly unlabeled future grid.")
    return zoo, fit, forecast


def run(team, output, origin, end, competition_test=False):
    zoo, fit, forecast = prepare_inputs(team, origin, end, competition_test)
    started = perf_counter()
    estimate = zoo.fit_xgb(fit, forecast)
    if not np.isfinite(estimate).all():
        raise ValueError("The frozen XGBoost model generated nonfinite estimates.")
    result = forecast[KEY].assign(origin=origin, xgb=estimate)
    temporary = output.with_name(output.stem + ".tmp.csv.gz")
    result.to_csv(temporary, index=False, float_format="%.8f")
    temporary.replace(output)
    print(f"Generated {len(result):,} fixed-origin XGBoost forecasts in {perf_counter() - started:.1f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--origin", default=ORIGIN)
    parser.add_argument("--end", default=END)
    parser.add_argument("--competition-test", action="store_true")
    args = parser.parse_args()
    run(args.team, args.output, args.origin, args.end, args.competition_test)
