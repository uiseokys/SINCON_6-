"""Validate non-game Munhak corrections from earlier errors and station peers."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sincon_overlay_candidate import replay, trained_inputs
from sincon_station_state import KEY
from sincon_weather_sensitivity import score
from sincon_xgb_conditioned import load_episodes


STADIUM = 3127
PEERS = (3126, 3128, 3129)
DAY = KEY[:2]
METHODS = ("historical_profile", "ordinary_residual", "ordinary_residual_with_peer")
WARM = (("2025-07-31", 8), ("2025-08-31", 9), ("2025-09-30", 10))
STADIUM_EVENTS = {
    "2025-07-27": "https://m.ticket.yes24.com/perf/detail/perfinfo.aspx?idperf=54411",
    "2025-08-10": "https://v.daum.net/v/20250806090234777",
    "2025-09-14": "https://www.bntnews.co.kr/article/view/bnt202509130066",
    "2025-10-24": "https://app.ssglanders.com/noticeView?seq=3074",
    "2026-05-30": "https://www.bntnews.co.kr/article/view/bnt202605300009",
    "2026-06-13": "https://www.bntnews.co.kr/article/view/bnt202605280155",
    "2026-06-20": "https://v.daum.net/v/20260620060717525",
    "2026-10-17": "https://www.ssglanders.com/media/news/detail?idx=19943",
    "2026-10-31": "https://m.ticket.yes24.com/Notice/18744?order=2&type=1",
}


def home_dates(work):
    with ZipFile(work / "sincon_local_repro.zip") as archive:
        schedule = pd.read_csv(archive.open("external_inputs/external/munhak_games.csv"),
                               parse_dates=["game_date"], usecols=["game_date", "status"])
    official = pd.read_csv(work / "sincon_kbo_game_data.csv", parse_dates=["date"], usecols=["date"])
    return pd.DatetimeIndex(schedule.game_date.unique()).union(official.date), {
        status: int(part.game_date.nunique()) for status, part in schedule.groupby("status")}


def category(frame, holidays, events, games):
    at_stadium = frame.station_id.eq(STADIUM)
    return np.select([at_stadium & frame.date.isin(games),
                      at_stadium & frame.date.isin(pd.to_datetime(list(STADIUM_EVENTS))),
                      pd.MultiIndex.from_frame(frame[DAY]).isin(events), frame.date.isin(holidays)],
                     ["kbo_home", "additional_stadium_event", "other_event", "calendar"], default="ordinary")


def robust_profile(history, target, origin):
    if history.date.max() > origin or target.date.min() <= origin or "observed" in target:
        raise ValueError("Local profiles require chronological observations and unlabeled target keys.")
    keys = ["date", "station_id"]
    past = history.pivot(index=keys, columns="hour", values="observed").sort_index()
    out = pd.Series(np.nan, index=pd.MultiIndex.from_frame(target[KEY]))
    for station, future in target.groupby("station_id", sort=False):
        observed = past.xs(station, level="station_id")
        dates, values = observed.index, observed.to_numpy()
        ages = (origin - dates).days.to_numpy()
        order = dates.day_of_week.to_numpy()
        year_order = dates.day_of_year.to_numpy()
        for day in future.date.unique():
            dow = day.day_of_week
            matching = order == dow
            type_match = (order >= 5) == (dow >= 5)
            same_season = np.exp(-2 * np.sin(np.pi * (year_order - day.day_of_year) / 365.25)**2)
            weight = np.exp2(-ages / 112) * (.25 + .75 * same_season)
            weight *= np.where(matching, 1., np.where(type_match, .18, 0.))
            median = np.median(values[matching], axis=0)
            robust = median + (values - median).clip(-.40, .40)
            profile = np.average(robust, weights=weight, axis=0)
            index = pd.MultiIndex.from_product([[day], [station], observed.columns], names=KEY)
            out.loc[index] = profile
    if out.isna().any():
        raise ValueError("Every target station-hour requires an earlier ordinary-day profile.")
    return out.to_numpy()


def episode_context(rows, history, origin, holidays, events, games):
    places = (STADIUM, *PEERS)
    target = rows.loc[rows.station_id.isin(places), [*KEY, "combined"]].reset_index(drop=True)
    fit = history.loc[history.date.le(origin) & history.station_id.isin(places)]
    fit = fit.loc[category(fit, holidays, events, games) == "ordinary"]
    if fit.date.max() > origin:
        raise ValueError("Ordinary training inputs crossed the forecast origin.")
    target["profile"] = robust_profile(fit, target, origin)
    target["category"] = category(target, holidays, events, games)
    peer_rows = target.loc[target.station_id.isin(PEERS) & target.category.eq("ordinary")]
    peer_difference = peer_rows.assign(change=(peer_rows.combined - peer_rows.profile)).groupby(DAY).change.mean()
    daily = peer_difference.groupby("date").agg(["median", "size"])
    median = daily["median"].where(daily["size"].ge(2), 0).clip(-.30, .30)
    main = target.loc[target.station_id.eq(STADIUM)].copy()
    main["peer"] = main.date.map(median).fillna(0)
    main["weekday"] = main.date.dt.day_of_week
    main["weekend"] = main.weekday.ge(5)
    return main


def offsets(past, future):
    numerator = past["weighted_error"]
    wide = past.assign(numerator=numerator).groupby(["weekend", "hour"])[["numerator", "weight"]].sum()
    pooled = wide.numerator / (wide.weight + 5)
    details = past.assign(numerator=numerator).groupby(["weekday", "hour"])[["numerator", "weight"]].sum()
    pooled = pooled.reindex(pd.MultiIndex.from_frame(future[["weekend", "hour"]])).fillna(0.).to_numpy()
    stats = details.reindex(pd.MultiIndex.from_frame(future[["weekday", "hour"]])).fillna(0.)
    return ((stats.numerator.to_numpy() + 6 * pooled) / (stats.weight.to_numpy() + 6)).clip(-.25, .25)


def historical_adjustment(future):
    shift = .30 * (future.profile.to_numpy() - future.combined.to_numpy()).clip(-.45, .45)
    return np.where(future.category.eq("ordinary"), shift, 0.)


def changes(episodes, future, origin):
    if future.date.min() <= origin or "y_log" in future:
        raise ValueError("Correction inputs must contain only held-out keys and their forecasts.")
    prior = episodes.loc[episodes.origin.lt(origin) & episodes.date.le(origin) & episodes.category.eq("ordinary")]
    prior = prior.sort_values(["reliability", "origin"], ascending=[False, True]).drop_duplicates(KEY).copy()
    ordinary = future.category.eq("ordinary").to_numpy()
    historical = historical_adjustment(future)
    residual, peer_delta = np.zeros(len(future)), np.zeros(len(future))
    beta = 0.
    days = int(prior.date.nunique())
    if days >= 30:
        prior["weight"] = prior.reliability.to_numpy() * np.exp2(-(origin - prior.date).dt.days.to_numpy() / 112)
        prior["error"] = (prior.y_log - prior.combined.clip(lower=0)).clip(-.45, .45)
        prior["weighted_error"] = prior.weight * prior.error
        residual = .6 * offsets(prior, future)
        prior["remaining"] = prior.error - offsets(prior, prior)
        station_days = prior.groupby("date").agg(peer=("peer", "first"), weight=("weight", "first"), error=("remaining", "mean"))
        x, w, y = (station_days[name].to_numpy() for name in ("peer", "weight", "error"))
        beta = float(np.clip(np.sum(w * x * y) / (np.sum(w * x**2) + .75), -.25, .25))
        peer_delta = beta * future.peer.to_numpy()
    return {name: np.where(ordinary, delta, 0.) for name, delta in zip(
        METHODS, (historical, residual, residual + peer_delta))}, {"days": days, "peer_coefficient": beta}


def metrics(rows):
    masks = {"all": np.ones(len(rows), dtype=bool), "station_3127": rows.station_id.eq(STADIUM).to_numpy(),
             "ordinary_3127": rows.category.eq("ordinary").to_numpy(),
             "ordinary_3127_late": (rows.category.eq("ordinary") & rows.hour.ge(21)).to_numpy()}
    return {group: {name: score(rows.y_log.to_numpy()[keep], rows.combined.to_numpy()[keep],
                               rows[name].to_numpy()[keep], rows.date[keep]) for name in METHODS}
            for group, keep in masks.items()}


def diagnosis(rows):
    station = rows.loc[rows.station_id.eq(STADIUM)].assign(
        error=lambda frame: frame.y_log - frame.combined.clip(lower=0),
    ).assign(square=lambda frame: frame.error**2)
    info = station.groupby("category").agg(rows=("square", "size"), days=("date", "nunique"), sse=("square", "sum"),
                                            mean_error=("error", "mean"))
    info["station_sse_share"] = info.sse / info.sse.sum()
    info["rmsle"] = np.sqrt(info.sse / info.rows)
    largest = (station.groupby(["date", "category"]).square.sum().nlargest(10).rename("sse").reset_index())
    regular = station.loc[station.category.eq("ordinary")].groupby("hour").agg(
        sse=("square", "sum"), mean_error=("error", "mean"))
    return {"categories": info.to_dict(orient="index"), "ordinary_hour_errors": regular.to_dict(orient="index"),
            "largest_station_days": json.loads(largest.to_json(orient="records", date_format="iso"))}


def run(args):
    episodes, raw, inputs, protected, calibrated, fallback = load_episodes(args)
    games, game_counts = home_dates(args.work_dir)
    fitted = trained_inputs(raw, args.work_dir, args.earlier, *protected)
    truth = raw.set_index(KEY).observed
    old = json.loads((args.work_dir / "sincon_overlay_candidate_results.json").read_text(encoding="utf-8"))
    full, archives, fits = {}, [], {}
    for name, (rows, origin, reliability) in sorted(episodes.items(), key=lambda value: value[1][1]):
        rows = rows.reset_index(drop=True)
        key = pd.MultiIndex.from_frame(rows[KEY])
        if not key.is_unique or not np.allclose(truth.reindex(key), rows.y_log, atol=1e-8, rtol=0):
            raise ValueError(f"Invalid official validation keys and labels in {name}.")
        exact, _ = replay(rows, origin, raw, inputs, protected, calibrated, fallback, fitted)
        actual = exact.y_log.to_numpy()
        baseline = exact.combined.to_numpy()
        if abs(np.sqrt(np.square(actual - np.maximum(baseline, 0)).mean()) - old[name]["all"]["combined"]["candidate"]) > 1e-10:
            raise ValueError(f"Exact incumbent replay differs in {name}.")
        context = episode_context(exact, raw, origin, *protected, games)
        prior = pd.concat(archives, ignore_index=True) if archives else context.iloc[:0].assign(origin=origin, reliability=1., y_log=0.)
        delta, fits[name] = changes(prior, context, origin)
        target = pd.MultiIndex.from_frame(context[KEY])
        case = exact[[*KEY, "y_log", "combined"]].copy()
        case["category"] = pd.Series(context.category.to_numpy(), index=target).reindex(key).fillna("outside_station").to_numpy()
        for method, difference in delta.items():
            adjustment = pd.Series(difference, index=target).reindex(key, fill_value=0.).to_numpy()
            if np.any(adjustment[~case.category.eq("ordinary")]) or not np.isfinite(adjustment).all():
                raise ValueError("Only ordinary days at station 3127 may change.")
            case[method] = baseline + adjustment
        full[name] = case
        archives.append(context.assign(origin=origin, reliability=reliability, y_log=truth.reindex(target).to_numpy()))
        print(name, fits[name], flush=True)
    full["disjoint_A_C"] = pd.concat([full["A"], full["C"]], ignore_index=True)
    full["disjoint_warm"] = pd.concat([full[name].loc[full[name].date.dt.month.eq(month)] for name, month in WARM], ignore_index=True)
    if any(full[name].duplicated(KEY).any() for name in ("disjoint_A_C", "disjoint_warm")):
        raise ValueError("Pooled folds must not share target rows.")
    return {"home_schedule_distinct_dates": len(games), "home_game_status_date_counts": game_counts,
            "additional_stadium_event_sources": STADIUM_EVENTS,
            "earlier_fits": fits, "cases": {name: {"scores": metrics(case), **diagnosis(case)} for name, case in full.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "earlier", "recent", "extra_xgb", "june_xgb", "april_xgb"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    started = perf_counter()
    report = run(args)
    report["seconds"] = round(perf_counter() - started, 2)
    (args.work_dir / "sincon_munhak_ordinary_results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
