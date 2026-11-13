"""Estimate Dongchun Sunday demand differences using pre-origin forecast residuals."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

STATION = 3131
KEY = ["date", "station_id", "hour"]
CUTOFF = pd.Timestamp("2026-06-30")
REPLAY = (Path(__file__).resolve().parent / "sincon_level_neutral_exact_replay.pkl")
INCUMBENT = "submission_station_profiles_event_program_level_neutral.csv"
SOURCE = "https://www.yeonsu.go.kr/main/general/civil/manless.asp"
CHECKSUMS = ("sincon_dongchun_retail_oof_delta.csv", "sincon_dongchun_retail_final_delta.csv")
CLOSED = ["2026-07-12", "2026-07-26", "2026-08-09", "2026-08-23", "2026-09-13", "2026-10-25"]
OPEN = ["2026-07-05", "2026-08-02", "2026-08-30", "2026-09-06", "2026-09-20", "2026-10-18"]
FLAGS = ["is_holiday", "is_bridge_day", "is_quasi_holiday", "is_pre_long_offday", "is_post_long_offday", "is_long_weekend"]
CASES = ("A", "B", "C", "summer", "2025-07-31", "2025-08-31", "2025-09-30", "2026-03-31", "2026-04-30")
BANDS = np.array([10, 16, 21, 23])
WEIGHT = .65


def is_closed(date):
    return date.dt.day.between(8, 14) | date.dt.day.between(22, 28)


def select(rows):
    return rows.loc[rows.station_id.eq(STATION) & rows.eligible & rows.date.dt.day_of_week.eq(6) & rows.hour.between(10, 22)].copy()


def contrasts(rows, weather=None):
    sample = rows.assign(closed=is_closed(rows.date), month=rows.date.dt.to_period("M"))
    by_band = sample.assign(band=np.searchsorted(BANDS, sample.hour, side="right")).groupby(
        ["date", "month", "closed", "band"], observed=True
    )[["error", "observed", "base"]].mean().reset_index()
    all_diffs = {}
    for band, part in by_band.groupby("band"):
        comparisons = []
        for month, month_days in part.groupby("month"):
            closures = month_days.loc[month_days.closed]
            opens = month_days.loc[~month_days.closed]
            if weather is not None and len(closures) and len(opens):
                closes_with_weather = closures.join(weather, on="date")
                opens_with_weather = opens.join(weather, on="date")
                distances = np.abs(closes_with_weather.day_temp_mean.to_numpy()[:, None] - opens_with_weather.day_temp_mean.to_numpy())
                agree = (closes_with_weather.day_precip_6_23.to_numpy()[:, None] > 5) == (opens_with_weather.day_precip_6_23.to_numpy() > 5)
                match = (distances <= 7) & agree
                if not match.any():
                    continue
                c_ids, o_ids = np.nonzero(match)
                a = closures.iloc[c_ids][["error", "observed", "base"]].to_numpy()
                b = opens.iloc[o_ids][["error", "observed", "base"]].to_numpy()
                comparisons.append((a - b).mean(axis=0))
            elif len(closures) and len(opens):
                comparisons.append(closures[["error", "observed", "base"]].mean().to_numpy()
                                   - opens[["error", "observed", "base"]].mean().to_numpy())
        if comparisons:
            all_diffs[band] = {"months": len(comparisons), **dict(zip(["error", "observed", "base"],
                                                                     np.mean(comparisons, axis=0).tolist()))}
    return all_diffs


def run_case(rows, evidence):
    sample = select(rows)
    delta = pd.Series(0.0, index=rows.index)
    weight = WEIGHT
    report = {}
    for band, hours in ((1, (10, 15)), (2, (16, 20)), (3, (21, 22))):
        fit = evidence.get(band, {})
        if fit.get("months", 0) < 2:
            continue
        positions = sample.index[sample.hour.between(*hours)]
        closed = is_closed(rows.loc[positions, "date"])
        contrast = np.clip(fit["error"], -.25, .25)
        delta.loc[positions] = weight * contrast * (closed.astype(float) - closed.mean())
        report[str(band)] = {"sampled_days": int(rows.loc[positions, "date"].nunique()),
                             "fit_contrast": fit["error"], "applied_contrast": weight * contrast}
    assert np.isclose(delta.sum(), 0.)
    return delta, report


def metrics(data):
    report = {}
    for name, part in (("all", data), ("station", data.loc[data.station_id.eq(STATION)]), ("affected", data.loc[data.delta.ne(0.)])):
        if part.empty:
            report[name] = {"n": 0}
            continue
        base = np.square(part.error).mean()**.5
        alt = np.square(part.error - part.delta).mean()**.5
        report[name] = {"n": len(part), "base": float(base), "candidate": float(alt), "gain": float(base-alt)}
    return report


def final_deltas(data, calendar, events):
    eligible_dates = pd.DatetimeIndex(pd.to_datetime(CLOSED + OPEN))
    if not eligible_dates.is_unique or not eligible_dates.day_of_week.isin([6]).all():
        raise ValueError("Verified retail calendar must uniquely list Sunday dates.")
    if calendar.reindex(eligible_dates)[FLAGS].isna().any().any():
        raise ValueError("Every verified date needs complete calendar flags.")
    eligible_dates = eligible_dates.difference(calendar.index[calendar[FLAGS].ne(0).any(axis=1)])
    eligible_dates = eligible_dates.difference(events.loc[events.station_id.eq(STATION), "date"])
    closed = pd.DatetimeIndex(pd.to_datetime(CLOSED))
    if not 0 < len(eligible_dates.intersection(closed)) < len(eligible_dates):
        raise ValueError("Both ordinary open and ordinary closed Sundays are required.")
    earlier = select(data.loc[data.date.le(CUTOFF) & data.origin.lt(CUTOFF)])
    earlier = earlier.sort_values(["reliability", "origin"]).drop_duplicates(KEY, keep="last")
    fit = contrasts(earlier)
    keys = pd.MultiIndex.from_product([eligible_dates, [STATION], range(10, 23)], names=KEY).to_frame(index=False)
    share = keys.date.isin(closed).mean()
    band = np.searchsorted(BANDS, keys.hour, side="right")
    contrast = pd.Series(band).map({k: WEIGHT * np.clip(v["error"], -.25, .25)
                                   for k, v in fit.items() if v["months"] >= 2})
    log_delta = contrast * (keys.date.isin(closed).astype(float) - share)
    if log_delta.isna().any() or not np.isclose(log_delta.sum(), 0.):
        raise ValueError("Expected finite and mean-neutral retail log offsets.")
    return keys.assign(log_delta=log_delta), {"fit": fit, "prior_sundays": int(earlier.date.nunique()),
             "closed": eligible_dates.intersection(closed).strftime("%Y-%m-%d").tolist(),
             "open": eligible_dates.difference(closed).strftime("%Y-%m-%d").tolist(), "rows": len(keys),
             "log_delta_min": float(log_delta.min()), "log_delta_max": float(log_delta.max())}


def run(data, calendar, events, weather):
    if data.date.max() > CUTOFF or not (data.date > data.origin).all() or data.duplicated(["label", *KEY]).any():
        raise ValueError("Expected unique strictly out-of-fold official train observations only.")
    data = data.assign(error=data.y_log - data.baseline, observed=data.y_log, base=data.baseline)
    ordinary = select(data)
    reports = {}
    oof_deltas = []
    for method in ("matched_month", "weather"):
        predictions, result = {}, {}
        for label in CASES:
            rows = data.loc[data.label.eq(label)]
            origin = rows.origin.iloc[0]
            prior = ordinary.loc[ordinary.date.le(origin) & ordinary.origin.lt(origin)]
            prior = prior.sort_values(["reliability", "origin"]).drop_duplicates(KEY, keep="last")
            matching = weather if method == "weather" else None
            evidence = contrasts(prior, weather=matching)
            delta, adjustments = run_case(rows[["eligible", *KEY]], evidence)
            result[label] = {
                "prior_sundays": int(prior.date.nunique()),
                "adjustment": adjustments,
                "score": metrics(rows.assign(delta=delta)),
            }
            predictions[label] = rows.assign(delta=delta)
            if method == "matched_month":
                changed = delta.ne(0.)
                oof_deltas.append(rows.loc[changed, ["label", "origin", *KEY]].assign(log_delta=delta.loc[changed]))
        for label, cases in {"A_C": [predictions["A"], predictions["C"]],
                             "Aug_Oct": [predictions[name].loc[predictions[name].date.dt.month.eq(month)]
                                         for name, month in zip(CASES[4:7], [8, 9, 10])]}.items():
            result[label] = {"score": metrics(pd.concat(cases))}
        reports[method] = result
    final, future = final_deltas(data, calendar, events)
    oof = pd.concat(oof_deltas, ignore_index=True).sort_values(["label", *KEY])
    if oof.duplicated(["label", *KEY]).any() or not np.isfinite(oof.log_delta).all():
        raise ValueError("OOF offsets must be finite and have unique keyed rows.")
    return {**reports, "final": future}, oof, final


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(work):
    with ZipFile(work / "sincon_local_repro.zip") as bundle:
        root = "external_inputs/"
        calendar = pd.read_csv(bundle.open(root + "calendar_features.csv"), parse_dates=["date"]).set_index("date")
        events = pd.read_csv(bundle.open(root + "external/events_daily.csv"), usecols=KEY[:2], parse_dates=["date"])
        mapping = pd.read_csv(bundle.open(root + "external/weather_zone_map.csv"))
        zone = mapping.loc[mapping.station_id.eq(STATION), "weather_zone"].item()
        weather = pd.read_csv(
            bundle.open(root + "external/weather_hourly.csv"),
            parse_dates=["date"],
            usecols=["date", "weather_zone", "hour", "day_temp_mean", "day_precip_6_23"],
        )
    festival = pd.read_csv(work / "sincon_festival_station_days.csv", usecols=KEY[:2], parse_dates=["date"])
    events = pd.concat([events, festival]).drop_duplicates()
    weather = weather.loc[weather.weather_zone.eq(zone) & weather.hour.eq(12)]
    weather = weather.set_index("date")[["day_temp_mean", "day_precip_6_23"]]
    return calendar, events, weather


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--team", type=Path, default=(Path(__file__).resolve().parent.parent / "team"))
    parser.add_argument("--replay", type=Path, default=REPLAY, help="Trusted local exact level-neutral validation archive")
    parser.add_argument("--incumbent", type=Path)
    args = parser.parse_args()
    work = args.work_dir
    incumbent = args.incumbent or work / INCUMBENT
    calendar, events, weather = inputs(work)
    data = pd.read_pickle(args.replay)
    train = pd.read_csv(args.team / "data/train.csv", usecols=[*KEY, "boardings"], parse_dates=["date"])
    truth = train.set_index(KEY).boardings
    if train.date.max() != CUTOFF or not truth.index.is_unique:
        raise ValueError("Expected unique official training labels ending June 30.")
    if not np.allclose(np.log1p(truth.reindex(pd.MultiIndex.from_frame(data[KEY]))), data.y_log, atol=1e-8, rtol=0):
        raise ValueError("Every replay label must match official training observations.")

    holiday_dates = calendar.index[calendar[FLAGS].ne(0).any(axis=1)]
    protected = data.date.isin(holiday_dates) | pd.MultiIndex.from_frame(data[KEY[:2]]).isin(
        pd.MultiIndex.from_frame(events[KEY[:2]])
    )
    data = data.assign(eligible=data.eligible & ~protected)
    report, oof_delta, final_delta = run(data, calendar, events, weather)
    grid = pd.read_csv(args.team / "data/test.csv", usecols=["example_id", *KEY], parse_dates=["date"])
    supplied = pd.read_csv(incumbent, usecols=["example_id", "boardings"])
    if not grid.example_id.equals(supplied.example_id) or not grid.example_id.is_unique:
        raise ValueError("The incumbent must have the official test IDs and order.")
    final_delta = grid.merge(final_delta, on=KEY, how="inner", validate="one_to_one")
    if len(final_delta) != report["final"]["rows"] or not final_delta.date.gt(CUTOFF).all():
        raise ValueError("The final offset must map exactly to its declared unlabelled keys.")
    oof_path, final_path = [work / name for name in CHECKSUMS]
    oof_delta.to_csv(oof_path, index=False, float_format="%.14f")
    final_delta[["example_id", *KEY, "log_delta"]].to_csv(final_path, index=False, float_format="%.14f")
    report["source_url"] = SOURCE
    report["incumbent"] = {"name": incumbent.name, "sha256": sha256(incumbent)}
    report["exact_replay_sha256"] = sha256(args.replay)
    report["artifacts"] = {path.name: {"rows": len(rows), "sha256": sha256(path)}
                           for path, rows in ((oof_path, oof_delta), (final_path, final_delta))}
    (work / "sincon_dongchun_retail_results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for label in ("A", "B", "C", "A_C", "Aug_Oct", "summer"):
        print(label, report["matched_month"][label]["score"])
    print("Final", report["final"], report["artifacts"])


if __name__ == "__main__":
    main()
