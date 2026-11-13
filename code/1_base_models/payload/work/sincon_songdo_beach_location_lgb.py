"""Audit the Songdo Beach venue and measure paired team LightGBM forecasts."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd


NEAREST_STATION = 3133
WRONG_STATION = 3139
STATIONS = (NEAREST_STATION, WRONG_STATION)
KEY = ["date", "station_id", "hour"]
LATE = (21, 22, 23)
HISTORICAL_ORIGIN = "2025-07-31"
FINAL_ORIGIN = "2026-06-30"
HISTORICAL_DAYS = pd.date_range("2025-08-09", "2025-08-15").strftime("%Y-%m-%d").tolist()
FINAL_DAYS = pd.date_range("2026-08-08", "2026-08-15").strftime("%Y-%m-%d").tolist()
PENTAPORT = ("2025-08-01", "2025-08-02", "2025-08-03")
SOURCES = {
    "host_confirmed_venue_and_visit_count": "https://www.yeonsu.go.kr/main/community/notify/report.asp?page=v&seq=29334",
    "host_directions_campus_town": "https://www.yeonsu.go.kr/tour/festival/beach/location.asp",
    "organizer_dates_and_program": "https://www.ysfac.or.kr/user/cultural/events_view.php?sq=381",
}


def event_removed(frame):
    absent = frame.copy()
    target = absent.station_id.eq(WRONG_STATION)
    if not absent.loc[target, "ev_today"].eq(1).all() or absent.loc[~target, "ev_today"].any():
        raise ValueError("Expected the original erroneous flags exclusively at station 3139.")
    absent.loc[target, ["ev_today", "ev_is_concert"]] = 0
    absent.loc[target, ["ev_day_index", "ev_n_days", "ev_hours_from_start"]] = np.nan
    if not absent.ev_city.equals(frame.ev_city):
        raise ValueError("A venue correction must preserve the citywide event count.")
    return absent


def add_forecasts(fit_predict, history, target, features, seeds):
    alternate = event_removed(target)
    observed, changed = np.split(np.log1p(fit_predict(history, pd.concat([target, alternate], ignore_index=True),
                                                      features, seeds)), 2)
    delta = changed - observed
    if not np.isfinite(delta).all() or np.any(delta[target.station_id.ne(WRONG_STATION)]):
        raise ValueError("Only the falsely flagged station may move.")
    return target[[*KEY, *(["example_id"] if "example_id" in target else [])]].assign(
        original_log_lgb=observed, no_false_flag_log_lgb=changed, delta_log=delta)


def scores(group):
    actual = group.y_log.to_numpy()
    candidates = {
        "archive": group.local_stack.to_numpy(),
        "archive_plus_half_paired_lgb_delta": group.local_stack.to_numpy() + .5 * group.delta_log.to_numpy(),
        "paired_lgb_original": group.original_log_lgb.to_numpy(),
        "paired_lgb_without_flag": group.no_false_flag_log_lgb.to_numpy(),
    }
    return {name: float(np.sqrt(np.square(actual - np.maximum(values, 0)).mean())) for name, values in candidates.items()}


def historical_summary(scored):
    groups = scored.groupby("station_id")
    return {str(station): {band: scores(part.loc[mask]) for band, mask in
                          (("all", np.ones(len(part), dtype=bool)), ("late", part.hour.isin(LATE)))}
            for station, part in groups}


def observed_and_archive(truth, archive, days):
    actual = truth.loc[truth.date.isin(days) & truth.station_id.isin(STATIONS), [*KEY, "boardings"]]
    panel = actual.merge(archive[[*KEY, "y_log", "local_stack"]], on=KEY, validate="one_to_one")
    if len(panel) != 18 * len(STATIONS) * len(days) or not np.allclose(np.log1p(panel.boardings), panel.y_log, atol=1e-8, rtol=0):
        raise ValueError("The official historical labels must match every archived key.")
    panel["local_stack_boardings"] = np.expm1(panel.local_stack)
    return panel


def daily_predictions(frame, columns):
    by_hour = frame.loc[frame.hour.isin(LATE)].pivot(index=["date", "station_id"], columns="hour", values=columns)
    by_hour.columns = [f"{column}_hour_{hour}" for column, hour in by_hour.columns]
    daily = frame.groupby(["date", "station_id"])[columns].sum().add_suffix("_all_day")
    return daily.join(by_hour).reset_index().to_dict(orient="records")


def evaluate(args):
    sys.path.insert(0, str(args.team.resolve()))
    from ensemble import FEATURES, LGB_SEEDS
    from improved import PARAMS, enrich, fit_predict
    from make_calendar import build_calendar

    PARAMS["num_threads"] = args.threads
    data = args.team / "data"
    training = pd.read_csv(data / "train.csv", usecols=[*KEY, "boardings"], dtype={"date": str})
    keys = pd.read_csv(data / "test.csv", usecols=[*KEY, "example_id"], dtype={"date": str})
    if (training.date.max() != FINAL_ORIGIN or len(keys) != 157_194 or keys.duplicated(KEY).any()
            or not keys.example_id.is_unique or "boardings" in keys):
        raise ValueError("Expected the official historical cutoff and unique unlabeled forecast keys.")
    meta = pd.read_csv(data / "metaData.csv", usecols=["station_id", "canonical_line"]).rename(columns={"canonical_line": "line"})
    calendar = build_calendar()
    train = enrich(training, calendar, meta, True)
    present = keys.loc[keys.date.isin(FINAL_DAYS) & keys.station_id.isin(STATIONS)]
    final = add_forecasts(fit_predict, train, enrich(present, calendar, meta, True), FEATURES, LGB_SEEDS)
    older = training.loc[training.date.isin(HISTORICAL_DAYS) & training.station_id.isin(STATIONS), KEY]
    prior = train.loc[train.date.le(HISTORICAL_ORIGIN)]
    historical = add_forecasts(fit_predict, prior, enrich(older, calendar, meta, True), FEATURES, LGB_SEEDS[:1])
    archived = pd.read_csv(args.rolling, dtype={"date": str, "origin": str},
                           usecols=[*KEY, "y_log", "local_stack", "origin"])
    if not archived.origin.eq(HISTORICAL_ORIGIN).all() or not historical.date.gt(HISTORICAL_ORIGIN).all():
        raise ValueError("Use only forecasts formed by July 31, 2025 for the historical comparison.")
    scored = observed_and_archive(training, archived, HISTORICAL_DAYS).merge(historical, on=KEY, validate="one_to_one")
    pentaport = observed_and_archive(training, archived, PENTAPORT)

    archived_lgb = pd.read_csv(args.team / "ensemble_parts.csv", usecols=["example_id", "log_lgb"])
    final = final.merge(archived_lgb, on="example_id", validate="one_to_one")
    fidelity = np.max(np.abs(final.original_log_lgb - final.log_lgb))
    if len(final) != 288 or fidelity > 1e-9:
        raise ValueError("The original final three-seed team forecasts must reproduce.")
    incumbent = pd.read_csv(args.incumbent, usecols=["example_id", "boardings"], float_precision="round_trip")
    final = final.merge(incumbent, on="example_id", validate="one_to_one")
    if len(final) != 288 or not np.isfinite(final.boardings).all() or final.boardings.lt(0).any():
        raise ValueError("Invalid incumbent predictions for final diagnostics.")
    final["half_weight_boardings"] = np.expm1((np.log1p(final.boardings) + .5 * final.delta_log).clip(lower=0.))
    final["full_delta_boardings"] = np.expm1((np.log1p(final.boardings) + final.delta_log).clip(lower=0.))
    return {
        "sources": SOURCES,
        "historical_origin": HISTORICAL_ORIGIN,
        "historical_scores": historical_summary(scored),
        "historical_beach_daily": daily_predictions(scored, ["boardings", "local_stack_boardings"]),
        "historical_pentaport_control": daily_predictions(pentaport, ["boardings", "local_stack_boardings"]),
        "final_archive_max_absolute_log_difference": float(fidelity),
        "final_changed_scope": final.loc[final.station_id.eq(WRONG_STATION)].to_dict(orient="records"),
        "final_daily_forecasts": daily_predictions(final, ["boardings", "half_weight_boardings", "full_delta_boardings"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team", type=Path)
    parser.add_argument("rolling", type=Path)
    parser.add_argument("incumbent", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--report", type=Path, default=Path("sincon_songdo_beach_location_lgb_results.json"))
    args = parser.parse_args()
    started = perf_counter()
    report = evaluate(args)
    report["seconds"] = round(perf_counter() - started, 2)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["historical_scores"], indent=2), flush=True)
