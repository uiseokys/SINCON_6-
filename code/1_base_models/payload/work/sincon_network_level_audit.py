"""Backtest chronological network-level alternatives to the released global offset."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_apply_final_calibration import PUBLIC_LOG_LEVEL
from sincon_asiad_concert_spillover import EVENTS as ASIAD_EVENTS, STATION as ASIAD_STATION, proposals as asiad_proposals
from sincon_asiad_science_festival import DATES as SCIENCE_DATES, STATIONS as SCIENCE_STATIONS
from sincon_munhak_ordinary import WARM, home_dates
from sincon_munhak_spectator import HISTORY, earlier_profiles
from sincon_overlay_candidate import replay, trained_inputs
from sincon_songdo_schedule import CHAMPION, save_submission
from sincon_spectator_month_split import gain_summary, month_halves
from sincon_station_profile_candidate import add_spectator_events, station_adjustments
from sincon_station_state import KEY
from sincon_weather_sensitivity import score
from sincon_xgb_conditioned import load_episodes


METHODS = ("without_public_level", "stable_common", "stable_day_type")
STABLE_EXCLUSIONS = (3107, 3108, 3109, 3110, 3111, 3127)


def ordinary(rows, protected):
    holidays, events = protected
    asiad = rows.station_id.eq(ASIAD_STATION) & rows.date.isin(ASIAD_EVENTS)
    science = rows.station_id.isin(SCIENCE_STATIONS) & rows.date.isin([date for pair in SCIENCE_DATES for date in pair])
    return ~rows.date.isin(holidays) & ~pd.MultiIndex.from_frame(rows[KEY[:2]]).isin(events) & ~asiad & ~science


def date_history(archive, origin, protected):
    window = archive.date.gt(origin - pd.Timedelta(days=140)) & archive.date.le(origin)
    past = archive.loc[window & archive.origin.lt(origin)].sort_values(["reliability", "origin"])
    past = past.drop_duplicates(KEY, keep="last")
    past = past.loc[ordinary(past, protected) & ~past.station_id.isin(STABLE_EXCLUSIONS)].copy()
    past["error"] = (past.y_log - (past.forecast + PUBLIC_LOG_LEVEL).clip(lower=0.)).clip(-.45, .45)
    dates = past.groupby("date").error.mean().rename("remaining_error").reset_index()
    return dates.assign(weekend=dates.date.dt.day_of_week.ge(5), month=dates.date.dt.to_period("M"))


def stable_shift(days, min_dates):
    monthly = days.groupby("month").remaining_error.agg(["median", "size"])
    monthly = monthly.loc[monthly["size"].ge(min_dates)]
    values = monthly["median"].to_numpy()
    pairs = {str(month): float(value) for month, value in monthly["median"].items()}
    if len(values) < 2 or not (np.all(values > 0.) or np.all(values < 0.)):
        return 0., pairs
    # Shrink toward the released value to limit an extrapolation from a few
    # months, and give each month's bias the same weight.
    return float(.5 * np.clip(np.median(values), -.06, .06)), pairs


def shifts(future, prior, origin, protected):
    if future.date.min() <= origin or {"y_log", "boardings"} & set(future):
        raise ValueError("Estimate calibration from older forecasts and apply it to unlabeled rows only.")
    dates = date_history(prior, origin, protected)
    common, common_evidence = stable_shift(dates, 14)
    weekday, weekday_evidence = stable_shift(dates.loc[~dates.weekend], 10)
    weekend, weekend_evidence = stable_shift(dates.loc[dates.weekend], 4)
    ordinary_scope = ordinary(future, protected) & ~future.station_id.isin(STABLE_EXCLUSIONS)
    day_type = np.where(future.date.dt.day_of_week.ge(5), weekend, weekday)
    result = {"without_public_level": np.where(ordinary_scope, -PUBLIC_LOG_LEVEL, 0.),
              "stable_common": np.where(ordinary_scope, common, 0.),
              "stable_day_type": np.where(ordinary_scope, day_type, 0.)}
    return result, {"prior_dates": len(dates), "remaining_error_by_month": common_evidence,
                    "weekday_error_by_month": weekday_evidence, "weekend_error_by_month": weekend_evidence,
                    "relative_common": common, "relative_weekday": weekday, "relative_weekend": weekend}


def metrics(rows):
    actual, base = rows.y_log.to_numpy(), rows.forecast.to_numpy() + PUBLIC_LOG_LEVEL
    return {method: {**score(actual, base, base + rows[method].to_numpy(), rows.date),
                     "changed_rows": int(rows[method].ne(0.).sum())} for method in METHODS}


def split_comparison(rows):
    actual, base = rows.y_log.to_numpy(), rows.forecast.to_numpy() + PUBLIC_LOG_LEVEL
    level_removed = base + rows.without_public_level.to_numpy()
    daily = rows[["date"]].assign(reference=(actual - base.clip(min=0.))**2,
                                  candidate=(actual - level_removed.clip(min=0.))**2, n=1).groupby("date").sum()
    half = month_halves(daily.index, 10_000, 116).astype(float)
    values = daily[["reference", "candidate"]].to_numpy()
    selected = half @ values
    selected_rows = half @ daily.n.to_numpy()
    complement_rows = daily.n.sum() - selected_rows
    left = np.sqrt(selected / selected_rows[:, None])
    right = np.sqrt((values.sum(axis=0) - selected) / complement_rows[:, None])
    paired = np.column_stack((left[:, 0] - left[:, 1], right[:, 0] - right[:, 1]))
    return {**gain_summary(paired.ravel()), "both_halves_improve_share": float(np.mean(np.all(paired > 0., axis=1)))}


def export(args, grid, dates, protected):
    if not args.baseline:
        raise ValueError("Export requires the verified latest forecast CSV.")
    keys = pd.read_csv(args.team / "data/test.csv", usecols=["example_id"])
    baseline = pd.read_csv(args.baseline, float_precision="round_trip")
    champion = pd.read_csv(args.team / CHAMPION, float_precision="round_trip")
    event_file = pd.read_csv(args.work_dir / "submission_local_stack_level_station_event_schedule.csv", float_precision="round_trip")
    if (len(keys) != 157_194 or not keys.example_id.is_unique or grid.duplicated(KEY).any()
            or not all(frame.example_id.equals(keys.example_id) for frame in (baseline, champion, event_file))
            or not np.isfinite(baseline.boardings).all() or baseline.boardings.lt(0.).any()):
        raise ValueError("The files must match the full official grid in order.")
    protected_events = event_file.boardings.ne(champion.boardings).to_numpy()
    if protected_events.sum() != 96:
        raise ValueError("Expected the 96 verified prior event rows.")
    scope = ordinary(grid, protected).to_numpy() & ~grid.station_id.isin(STABLE_EXCLUSIONS).to_numpy()
    if int(scope.sum()) != dates:
        raise ValueError("The test sensitivity must match the actual ordinary mature rows.")
    apply = scope & ~protected_events
    new = baseline.boardings.to_numpy(copy=True)
    new[apply] = np.expm1((np.log1p(new[apply]) - PUBLIC_LOG_LEVEL).clip(min=0.))
    save_submission(args.submission, baseline, keys, new)
    return {"submission": str(args.submission.resolve()), "rows": len(keys), "changed": int(np.count_nonzero(new != baseline.boardings)),
            "verified_event_rows_preserved": int(protected_events.sum()),
            "removed_log_offset": PUBLIC_LOG_LEVEL, "sha256": hashlib.sha256(args.submission.read_bytes()).hexdigest()}


def evaluate(args):
    episodes, raw, inputs, protected, calibrated, fallback = load_episodes(args)
    fitted = trained_inputs(raw, args.work_dir, args.earlier, *protected)
    games, _ = home_dates(args.work_dir)
    event_profiles = earlier_profiles(raw, protected, games)
    official = raw.set_index(KEY).observed
    forecasts, archives = {}, []
    for label, (rows, origin, reliability) in episodes.items():
        rows = rows.reset_index(drop=True)
        index = pd.MultiIndex.from_frame(rows[KEY])
        if not index.is_unique or rows.date.min() <= origin or not np.allclose(official.reindex(index), rows.y_log, atol=1e-8, rtol=0.):
            raise ValueError(f"Invalid official labels or chronological keys in {label}.")
        base, _ = replay(rows, origin, raw, inputs, protected, calibrated, fallback, fitted)
        query = base[[*KEY, "combined"]]
        delta, _ = station_adjustments(raw.loc[raw.date.le(origin)], query, origin, protected, games)
        delta, _ = add_spectator_events(delta, query, HISTORY, origin, event_profiles)
        spectator = base.combined + delta["joint_with_spectator"]
        concert, _ = asiad_proposals(base[KEY].assign(incumbent=spectator), raw.loc[raw.date.le(origin)],
                                     origin, protected, include_placebos=False)
        forecasts[label] = base[KEY].assign(y_log=base.y_log, forecast=concert.three_quarters_excess)
        archives.append(forecasts[label].assign(origin=origin, reliability=reliability))
    archive = pd.concat(archives, ignore_index=True)
    case, report = {}, {}
    for label, frame in forecasts.items():
        origin = episodes[label][1]
        changes, support = shifts(frame[KEY], archive, origin, protected)
        case[label] = frame.assign(**changes)
        report[label] = {"origin": str(origin.date()), "support": support, **metrics(case[label])}
    warm = pd.concat([case[label].loc[case[label].date.dt.month.eq(month)] for label, month in WARM], ignore_index=True)
    ac = pd.concat([case[label] for label in ("A", "C")], ignore_index=True)
    for label, subset in (("disjoint_A_C", ac), ("disjoint_warm", warm)):
        if subset.duplicated(KEY).any():
            raise ValueError("Each pooled comparison must use distinct dates.")
        report[label] = metrics(subset)
        report[label]["without_public_level_month_halves"] = split_comparison(subset)
    grid = pd.read_csv(args.team / "data/test.csv", parse_dates=["date"], usecols=KEY)
    potential, support = shifts(grid, archive, pd.Timestamp("2026-06-30"), protected)
    report["test_sensitivity"] = {"support": support, **{method: {"rows": int(np.count_nonzero(values)),
                                                                  "mean": float(np.mean(values))}
                                                         for method, values in potential.items()}}
    if args.submission:
        report["export"] = export(args, grid, report["test_sensitivity"]["without_public_level"]["rows"], protected)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "earlier", "recent", "extra_xgb", "june_xgb", "april_xgb"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    report = evaluate(args)
    (args.work_dir / "sincon_network_level_audit_results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("A", "B", "C", "disjoint_A_C", "disjoint_warm", "test_sensitivity")}), flush=True)


if __name__ == "__main__":
    main()
