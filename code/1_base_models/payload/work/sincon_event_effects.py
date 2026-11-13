"""Compare a shared event-count effect with venue and recurring-event profiles."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

import sincon_external_covariates as external
import sincon_forecast as forecast
from sincon_lightgbm_experiments import HOLIDAY_FLAGS, attach_calendar, holiday_snapshot


SERIES = ("펜타포트", "맥주축제", "해변축제", "능허대", "흠뻑쇼", "CARAT LAND", "박효신")
PROFILE_KEYS = ["station_id", "date"]


def read_events(path):
    columns = ["date", "station_id", "event_name", "category", "event_day_index", "event_n_days"]
    with ZipFile(path) as archive:
        events = pd.read_csv(archive.open("external/events_daily.csv"), usecols=columns, parse_dates=["date"])
    events = events.loc[events["station_id"].gt(0)].copy()
    events["series"] = events["event_name"].map(lambda name: next((key for key in SERIES if key in name), name))
    events["event_start"] = events["date"] - pd.to_timedelta(events["event_day_index"] - 1, unit="D")
    return events


def estimate_profiles(history, events, games):
    known = events.loc[events["date"].le(history["date"].max())]
    logged = history.assign(log_count=np.log1p(history["boardings"].to_numpy(dtype="float64")))
    ordinary = logged.join(known.groupby(PROFILE_KEYS).size().rename("events"), on=PROFILE_KEYS)
    effects = []
    for station, frame in ordinary.loc[ordinary["events"].isna()].groupby("station_id"):
        station_events = known.loc[known["station_id"].eq(station)].drop_duplicates("date", keep=False)
        if station_events.empty:
            continue
        daily = frame.pivot(index="date", columns="hour", values="log_count")
        groups = frame.drop_duplicates("date").set_index("date")["day_group"].reindex(daily.index)
        observed = logged.loc[logged["station_id"].eq(station)].pivot(index="date", columns="hour", values="log_count")
        source_groups = logged.loc[logged["station_id"].eq(station)].drop_duplicates("date").set_index("date")["day_group"]
        for event in station_events.itertuples(index=False):
            if event.date not in observed.index:
                continue
            same_day = groups.eq(source_groups.loc[event.date])
            distance = np.abs((daily.index - event.date).days)
            matches = same_day & (distance <= 90)
            if station == external.STADIUM:
                played = games["games_played"].gt(0)
                matches &= played.reindex(daily.index).eq(played.loc[event.date])
            if matches.sum() >= 4:
                normal = daily.loc[matches].median(axis=0).to_numpy()
                effects.append((event.station_id, event.event_start, event.category, event.series,
                                event.event_n_days, observed.loc[event.date].to_numpy() - normal))
    # One profile per event so a long festival cannot dominate the prior.
    per_event = {}
    for *key, hourly in effects:
        per_event.setdefault(tuple(key), []).append(hourly)
    return [(key, np.median(profiles, axis=0)) for key, profiles in per_event.items()]


def uplift_for_event(event, profiles, shared, scaled):
    if not scaled:
        return shared
    recurrence = [profile for (station, _, _, series, _), profile in profiles
                  if station == event.station_id and series == event.series]
    if recurrence:
        return 0.85 * np.median(recurrence, axis=0)
    local = [profile for (station, _, category, _, duration), profile in profiles
             if station == event.station_id and category == event.category
             and 0.5 <= duration / event.event_n_days <= 2]
    if local:
        return (0.60 + 0.20 * len(local) / (len(local) + 2)) * np.median(local, axis=0)
    # Limited transfer to an unseen venue or a different event format.
    category = [profile for (_, _, kind, _, _), profile in profiles if kind == event.category]
    return 0.25 * np.median(category, axis=0) if category else np.zeros_like(shared)


def event_forecasts(history, future, events, games, reference):
    profiles = estimate_profiles(history, events, games)
    shared = 0.7 * np.median([profile for _, profile in profiles], axis=0) if profiles else np.zeros(18)
    target = events.loc[events["date"].isin(future["date"])]
    base = pd.Series(reference, index=pd.MultiIndex.from_frame(future[forecast.KEY]))
    additions = {"event_count": pd.Series(0., index=base.index),
                 "event_venue_and_series": pd.Series(0., index=base.index)}
    for event in target.itertuples(index=False):
        keys = pd.MultiIndex.from_product([[event.date], [event.station_id], external.HOURS], names=forecast.KEY)
        existing = base.reindex(keys)
        if existing.isna().any():
            continue
        for name, scaled in [("event_count", False), ("event_venue_and_series", True)]:
            log_uplift = np.clip(uplift_for_event(event, profiles, shared, scaled), 0, 4)
            extra = (1 + existing.to_numpy()) * np.expm1(log_uplift)
            additions[name].loc[keys] += extra
    return {name: (base + extra).to_numpy() for name, extra in additions.items()}, len(profiles)


def run(data_zip, calendar_csv, external_zip, threads, submission=None):
    train, test = forecast.read_inputs(data_zip)
    dates = pd.date_range(train["date"].min(), test["date"].max(), name="date")
    calendar = pd.read_csv(calendar_csv, usecols=["date", *HOLIDAY_FLAGS], parse_dates=["date"]).set_index("date").sort_index()
    weather, games = external.read_external(external_zip, dates)
    events = read_events(external_zip)
    results = {}
    for fold, cutoff in forecast.BACKTESTS.items():
        history, validation = forecast.split_backtest(train, cutoff)
        future = validation.drop(columns="boardings")
        baseline, _ = external.predict_at(history, future, calendar, weather, games, cutoff, threads)
        reference = baseline["all_external"]
        historical = attach_calendar(history, holiday_snapshot(calendar, cutoff), cutoff)
        variants, fitted_events = event_forecasts(historical, future, events, games, reference)
        variants["weather_and_baseball"] = reference
        event_days = pd.MultiIndex.from_frame(events[PROFILE_KEYS])
        in_event = pd.MultiIndex.from_frame(validation[PROFILE_KEYS]).isin(event_days)
        results[fold] = {"rows": len(validation), "fitted_events": fitted_events, "event_rows": int(in_event.sum()),
                         "rmsle": {name: forecast.rmsle(validation["boardings"], values) for name, values in variants.items()},
                         "event_rmsle": {name: forecast.rmsle(validation.loc[in_event, "boardings"], values[in_event])
                                         for name, values in variants.items()}}
        print(json.dumps({fold: results[fold]}, allow_nan=False), flush=True)
    pooled = {name: float(np.sqrt(np.average([score["rmsle"][name]**2 for score in results.values()],
                                             weights=[score["rows"] for score in results.values()]))) for name in variants}
    event_pooled = {name: float(np.sqrt(np.average([score["event_rmsle"][name]**2 for score in results.values()],
                                                   weights=[score["event_rows"] for score in results.values()])))
                    for name in variants}
    final = None
    if submission:
        cutoff = forecast.CUTOFF
        baseline, _ = external.predict_at(train, test, calendar, weather, games, cutoff, threads)
        historical = attach_calendar(train, holiday_snapshot(calendar, cutoff), cutoff)
        variants, fitted_events = event_forecasts(historical, test, events, games, baseline["all_external"])
        values = variants["event_venue_and_series"]
        forecast.write_submission(forecast.make_submission(test, values), submission)
        final = {"rows": len(test), "fitted_events": fitted_events,
                 "adjusted_rows": int(np.count_nonzero(values != baseline["all_external"]))}
    return {"folds": results, "pooled_rmsle": pooled, "event_pooled_rmsle": event_pooled,
            "status": "exploratory_event_list_selected_with_training_data", "final": final}


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
    print(json.dumps({key: output[key] for key in ("pooled_rmsle", "event_pooled_rmsle", "final")}, allow_nan=False), flush=True)
