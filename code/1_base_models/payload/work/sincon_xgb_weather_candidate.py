"""Evaluate a fixed XGBoost blend and optional late-hour weather correction."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_complementary_blend import historical_station_shift
from sincon_songdo_schedule import CHAMPION, aligned_submission, save_submission
from sincon_station_state import KEY, WINDOWS
from sincon_team_bias_proxy import earlier_team_errors
from sincon_weather_sensitivity import context, forecast, score
from sincon_weekly_variability import past_episodes, protected_days
from sincon_zoo_transfer import SEASON_WEIGHT, features, verified_archive


WEATHER_WEIGHT = .45
EVENT_CANDIDATE = "submission_local_stack_level_station_event_schedule.csv"


def alternate_archive(local, alternate, origin):
    fields = [*KEY, "local_stack", "y_log"]
    base = pd.read_csv(local, parse_dates=["date"], usecols=fields)
    xgb = pd.read_csv(alternate, parse_dates=["date", "origin"], usecols=[*KEY, "xgb", "origin"])
    rows = base.merge(xgb, on=KEY, validate="one_to_one", how="outer", indicator=True)
    if not rows.pop("_merge").eq("both").all() or not rows.origin.eq(origin).all():
        raise ValueError(f"The validation forecasts must have complete keys and the same origin: {alternate}")
    return rows


def adjustments(rows, raw, origin, inputs, protected):
    if not np.isfinite(rows[["local_stack", "xgb"]]).all().all():
        raise ValueError("The local and XGBoost forecasts must be finite.")
    ordinary, xgb, _ = features(rows, ["xgb"], *protected)
    weather = forecast(raw.loc[raw.date.le(origin)], rows[[*KEY, "local_stack"]], origin, inputs)
    late = rows.hour.ge(21) & ordinary
    weather = WEATHER_WEIGHT * np.where(late, weather, 0.)
    xgb = SEASON_WEIGHT * xgb.xgb.to_numpy()
    mature = rows.station_id.map(inputs[0].first_date).eq(inputs[0].first_date.min()).to_numpy()
    return {
        "xgb": xgb,
        "xgb_weather": xgb + weather,
        "mature_xgb_weather": xgb * mature + weather,
        "late_weather": weather,
    }


def segment_scores(rows, variants, station):
    reference = rows.reference.to_numpy()
    actual = rows.y_log.to_numpy()
    date = rows.date
    mature = rows.station_id.map(station.first_date).eq(station.first_date.min()).to_numpy()
    masks = {"all": np.ones(len(rows), dtype=bool), "mature_68": mature, "late_21_23": rows.hour.ge(21).to_numpy()}
    return {scope: {name: score(actual[mask], reference[mask], rows[name].to_numpy()[mask], date[mask])
                    for name in variants} for scope, mask in masks.items()}


def validate(args, raw, inputs, protected):
    work = args.work_dir
    original, _ = verified_archive(work, args.team)
    columns = [*KEY, "team", "y_log"]
    team_main = pd.read_csv(work / "sincon_local_validation.csv.gz", usecols=[*columns, "fold"], parse_dates=["date"])
    team_summer = pd.read_csv(work / "sincon_summer_replay_validation.csv.gz", usecols=columns, parse_dates=["date"])
    origins = {name: pd.Timestamp(span[0]) for name, span in WINDOWS.items()}
    june, april = pd.Timestamp("2025-06-30"), pd.Timestamp("2026-04-30")
    history = pd.concat([team_main.assign(origin=team_main.fold.map(origins)), team_summer.assign(origin=june)], ignore_index=True)
    history["station_age"] = (history.origin - history.station_id.map(inputs[0].first_date)).dt.days
    _, calibration = past_episodes(work / "sincon_local_validation.csv.gz", args.earlier, *protected)
    episodes = {name: (original.loc[original.fold.eq(name)].copy(), origin) for name, origin in origins.items()}
    episodes["summer"] = (alternate_archive(work / "sincon_summer_replay_validation.csv.gz", args.season_xgb, june), june)
    episodes["recent"] = (alternate_archive(args.april_local, args.april_xgb, april), april)
    official = raw.set_index(KEY).observed
    report, collected = {}, {}
    for name, (rows, origin) in episodes.items():
        labels = official.reindex(pd.MultiIndex.from_frame(rows[KEY])).to_numpy()
        if rows.date.min() <= origin or not np.allclose(labels, rows.y_log, rtol=0, atol=1e-8):
            raise ValueError(f"Invalid fixed-origin labels in {name}.")
        changes = adjustments(rows, raw, origin, inputs, protected)
        offsets, days = earlier_team_errors(history, origin)
        if days >= 60:
            fallback, _ = historical_station_shift(calibration, rows, origin)
            shift = rows.station_id.map(offsets).fillna(pd.Series(fallback, index=rows.index)).to_numpy()
        else:
            shift = 0.
        reference = rows.local_stack.to_numpy() + shift
        predictions = {key: reference + change for key, change in changes.items()}
        scored = rows[[*KEY, "y_log"]].assign(reference=reference, **predictions)
        collected[name] = scored
        report[name] = {"origin": str(origin.date()), "calibration_days": days, **segment_scores(scored, changes, inputs[0])}
        if name in ("summer", "recent"):
            report[name]["months"] = {str(month): segment_scores(group, changes, inputs[0])
                                       for month, group in scored.groupby(scored.date.dt.to_period("M"))}
    pooled = pd.concat([collected["A"], collected["C"]], ignore_index=True)
    if pooled.duplicated(KEY).any():
        raise ValueError("The pooled A/C comparison requires distinct days.")
    report["disjoint_A_C"] = segment_scores(pooled, changes, inputs[0])
    return report


def write_final(args, raw, inputs, protected):
    work = args.work_dir
    origin = pd.Timestamp("2026-06-30")
    test = pd.read_csv(args.team / "data/test.csv", parse_dates=["date"], usecols=[*KEY, "example_id"])
    if len(test) != 157_194 or not test.example_id.is_unique or test.duplicated(KEY).any() or raw.date.max() != origin:
        raise ValueError("The final forecast requires the official 157,194 test keys and June 30 cutoff.")
    xgb = pd.read_csv(args.final_xgb, parse_dates=["date", "origin"], usecols=[*KEY, "xgb", "origin"])
    base = aligned_submission(work / "submission_local_stack.csv", test)
    rows = test[KEY].assign(local_stack=base).merge(xgb, on=KEY, how="left", validate="one_to_one")
    if not rows.origin.eq(origin).all() or len(xgb) != len(test) or not rows[KEY].equals(test[KEY]):
        raise ValueError("The XGBoost forecast must have the official final keys and origin.")
    changes = adjustments(rows, raw, origin, inputs, protected)
    reference = pd.read_csv(work / EVENT_CANDIDATE, float_precision="round_trip")
    champion = pd.read_csv(args.team / CHAMPION, float_precision="round_trip")
    if not reference.example_id.equals(test.example_id) or not champion.example_id.equals(test.example_id):
        raise ValueError("The released prediction files must match test.csv order.")
    protected_events = reference.boardings.ne(champion.boardings).to_numpy()
    delta = np.where(protected_events, 0., changes[args.variant])
    values = reference.boardings.to_numpy(copy=True)
    active = np.abs(delta) > 1e-12
    values[active] = np.expm1(np.maximum(np.log1p(values[active]) + delta[active], 0.))
    if protected_events.sum() != 96 or np.any(values[protected_events] != reference.boardings[protected_events]):
        raise ValueError("The original 96 separately checked event updates must be preserved.")
    if args.submission:
        save_submission(args.submission, reference, test, values)
    return {"rows": len(values), "variant": args.variant, "changed_vs_event_candidate": int(active.sum()),
            "protected_event_rows": int(protected_events.sum()), "log_change_rms": float(np.sqrt(np.mean(delta**2))),
            "log_change_max": float(np.abs(delta).max()), "submission": str(args.submission.resolve()) if args.submission else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "earlier", "season_xgb", "april_local", "april_xgb"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--final-xgb", type=Path)
    parser.add_argument("--variant", choices=("xgb", "xgb_weather", "mature_xgb_weather"), default="xgb_weather")
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    if args.submission and not args.final_xgb:
        parser.error("The official XGBoost forecast is required to export the submission.")
    raw = pd.read_csv(args.team / "data/train.csv", usecols=[*KEY, "boardings"], parse_dates=["date"])
    raw["observed"] = np.log1p(raw.pop("boardings"))
    inputs = context(args.work_dir, args.team)
    protected = protected_days(args.work_dir)
    result = validate(args, raw, inputs, protected)
    if args.final_xgb:
        result["final"] = write_final(args, raw, inputs, protected)
    path = args.work_dir / "sincon_xgb_weather_candidate_results.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: section.get("all", section) for name, section in result.items()}), flush=True)


if __name__ == "__main__":
    main()
