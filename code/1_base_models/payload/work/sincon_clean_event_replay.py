"""Rebuild the event layers on the uncalibrated forecast and clean historical predictions."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd

import sincon_asiad_concert_spillover as asiad
import sincon_dreampark_festival_audit as dream
import sincon_kbo_cancellation_time as rain
import sincon_munhak_cancellation_candidate as munhak_cancel
import sincon_munhak_spectator as spectator
import sincon_neunghudae_festival_audit as neunghudae
import sincon_songdo_beach_location_candidate as beach_location
import sincon_trail_festival_audit as trail
import sincon_wonmi_namdong_audit as wonmi
from sincon_munhak_ordinary import category, home_dates
from sincon_public14774_event_followup import DATE_PAIRS, earlier_controls, excess, station_profiles
from sincon_public14774_event_update import digest, submission
from sincon_songdo_beach_evening_audit import TRANSFER, paired as beach_evening
from sincon_songdo_schedule import save_submission
from sincon_station_state import KEY
from sincon_weekly_variability import protected_days


ORIGIN = pd.Timestamp("2026-06-30")
GRID = ["example_id", *KEY]
SCIENCE = "sincon_asiad_science_festival_results.json"


def inputs(team, work, baseline):
    grid = pd.read_csv(team / "data/test.csv", usecols=GRID, parse_dates=["date"])
    history = pd.read_csv(team / "data/train.csv", usecols=[*KEY, "boardings"], parse_dates=["date"])
    if (len(grid) != 157_194 or grid.duplicated(KEY).any() or not grid.example_id.is_unique
            or history.date.max() != ORIGIN or len(history) != 688_176 or history.duplicated(KEY).any()):
        raise ValueError("Expected the official train.csv and complete test.csv.")
    grid["base"] = submission(baseline, grid).boardings
    history["observed"] = np.log1p(history.boardings)
    return grid, history, protected_days(work)


def initial_event_keys(grid):
    from sincon_songdo_schedule import END, START, STATIONS

    songdo = grid.station_id.isin(STATIONS) & grid.date.between(START, END) & grid.hour.isin((21, 22))
    pungmul = grid.date.eq("2026-10-02") & grid.station_id.isin((3119, 3120)) & grid.hour.ge(21)
    bicof = grid.date.between("2026-09-18", "2026-09-20") & grid.station_id.eq(3759)
    selected = grid.loc[songdo | pungmul | bicof, GRID]
    if len(selected) != 96:
        raise ValueError("All original event adjustments must remain protected.")
    return selected


def update(grid, selection, logs, label):
    selected = grid.loc[selection, GRID + ["base"]]
    if not len(selected) or not np.isfinite(logs).all() or len(logs) != len(selected):
        raise ValueError(f"Invalid adjustments for {label}.")
    values = np.expm1(np.maximum(logs, 0.))
    before = selected.base.to_numpy()
    changed = np.abs(before - values) > 1e-9
    grid.loc[selected.index[changed], "base"] = values[changed]
    return selected.loc[changed].assign(after=values[changed], layer=label)


def add_offsets(grid, offsets, label):
    offsets = offsets.set_index("example_id", verify_integrity=True)
    mask = grid.example_id.isin(offsets.index)
    lookup = offsets.reindex(grid.loc[mask, "example_id"])
    if len(lookup) != len(offsets):
        raise ValueError(f"{label} keys were lost.")
    return update(grid, mask, np.log1p(grid.loc[mask, "base"].to_numpy()) + lookup.log_delta.to_numpy(), label)


def paired_removal(team, work, grid, temporary):
    pairs = [
        ("munhak_cancellation", "sincon_munhak_cancellation_responses.zip",
         "sincon_munhak_cancellation_lgb_results.json", "sincon_munhak_cancelled_nn_seed", munhak_cancel.run, 72),
        ("songdo_beach_location", "sincon_songdo_beach_location_responses.zip",
         "sincon_songdo_beach_location_lgb_results.json", "sincon_songdo_beach_nn_seed", beach_location.evaluate, 24),
    ]
    changes = []
    for label, archive, tree, neural, predict, expected in pairs:
        folder = temporary / label
        folder.mkdir()
        with ZipFile(work / archive) as source:
            for name in [tree, *[f"{neural}{seed}.csv" for seed in range(3)]]:
                (folder / name).write_bytes(source.read(name))
        base = folder / "clean_input.csv"
        save_submission(base, grid[["example_id"]], grid, grid.base.to_numpy())
        scenario = argparse.Namespace(team=team, baseline=base, tree=folder / tree, submission=None,
                                      neural=[folder / f"{neural}{seed}.csv" for seed in range(3)])
        report = predict(scenario)
        records = pd.DataFrame(report["hourly"])
        effect = .5 * (records.lgb_log_delta + records.nn_log_delta)
        if len(records) != expected:
            raise ValueError(f"Unexpected paired-response coverage for {label}.")
        changes.append(add_offsets(grid, records.assign(log_delta=effect), label))
    response = beach_evening(work, grid.assign(date=grid.date.dt.strftime("%Y-%m-%d")), team)
    changes.append(add_offsets(grid, response.assign(log_delta=TRANSFER * response.paired), "songdo_beach_evening"))
    return changes


def historical_events(grid, history, protected, work):
    games, _ = home_dates(work)
    observed = history[[*KEY, "observed"]]
    profiles = spectator.earlier_profiles(observed, protected, games)
    changes = []
    for game in spectator.FINAL:
        selected = grid.date.eq(game.date) & grid.station_id.eq(spectator.STADIUM)
        rows = grid.loc[selected].sort_values("hour")
        if not np.array_equal(rows.hour, spectator.HOURS):
            raise ValueError("Expected all eighteen spectator-event hours.")
        preds, precedents = spectator.proposals(game, ORIGIN, np.log1p(rows.base.to_numpy()), profiles)
        if not precedents:
            raise ValueError("The spectator adjustment needs past event support.")
        changes.append(update(grid, rows.index, preds["fixed"], "munhak_spectator"))

    future = grid[KEY].assign(incumbent=np.log1p(grid.base))
    estimates, _ = asiad.proposals(future, observed, ORIGIN, protected, include_placebos=False)
    selected = grid.date.eq("2026-07-11") & grid.station_id.eq(3210) & grid.hour.isin((21, 22))
    changes.append(update(grid, selected, estimates.loc[selected, "three_quarters_excess"].to_numpy(), "asiad_psy"))

    stadium = observed.loc[observed.station_id.eq(spectator.STADIUM)]
    ordinary = stadium.loc[category(stadium, *protected, games) == "ordinary"]
    profiles = rain.build_history(stadium, ordinary)
    for event in rain.FINAL:
        selected = grid.date.eq(event.date) & grid.station_id.eq(spectator.STADIUM)
        rows = grid.loc[selected, KEY].assign(forecast=np.log1p(grid.loc[selected, "base"])).sort_values("hour")
        pred, donors = rain.proposal(event, rows, ORIGIN, ordinary, profiles)
        if not donors:
            raise ValueError("Rain cancellation forecasts need past precedents.")
        changes.append(update(grid, rows.index, pred, "munhak_rain_cancellation"))
    return changes


def science(grid, history, work, refine):
    source = json.loads((work / SCIENCE).read_text())
    shifts = []
    for station in (3209, 3210):
        actual = station_profiles(history, station, "boardings")
        predictions = station_profiles(grid, station, "base")
        weights = {15: .5, 16: .5, 17: .5}
        if refine:
            weights = {16: .75, 17: .75}
            if station == 3210:
                weights.update({14: .25, 18: .25})
        for prior, day in DATE_PAIRS:
            control = source["unvalidated_projection_only"][f"{station}:{day.date()}"]
            if control["source"] != str(prior.date()):
                raise ValueError("Science festival dates do not match the documented previous editions.")
            anchors = pd.DatetimeIndex(control["unlabelled_control_dates"])
            if not (anchors < day).all() or not (anchors.day_of_week == day.day_of_week).all():
                raise ValueError("Use earlier same-weekday forecasts as science-festival controls.")
            floors = predictions.loc[anchors].median().reindex(weights) + pd.Series(weights) * excess(actual, source, station, prior).reindex(weights)
            selected = grid.station_id.eq(station) & grid.date.eq(day) & grid.hour.isin(weights)
            levels = np.maximum(grid.loc[selected, "base"], grid.loc[selected, "hour"].map(floors))
            shifts.append(update(grid, selected, np.log1p(levels.to_numpy()), "science_refinement" if refine else "science"))
    return shifts


def clean_history(path, truth):
    validation = pd.read_csv(path, usecols=[*KEY, "label", "origin", "baseline"], parse_dates=["date", "origin"])
    observed = truth.set_index(KEY).observed
    y = observed.reindex(pd.MultiIndex.from_frame(validation[KEY]))
    if (y.isna().any() or not validation.date.gt(validation.origin).all() or not np.isfinite(validation.baseline).all()
            or validation.duplicated(["label", *KEY]).any()):
        raise ValueError("Historical event calibration must use aligned clean OOF and official labels.")
    return validation.assign(y_log=y.to_numpy())


def wonmi_adjustment(grid, history, archive, protected, work):
    with ZipFile(work / "sincon_local_repro.zip") as bundle:
        root = "external_inputs/external/"
        weather = pd.read_csv(bundle.open(root + "weather_hourly.csv"), parse_dates=["date"],
                              usecols=["weather_zone", "date", "hour", "temp", "precip"])
        zones = pd.read_csv(bundle.open(root + "weather_zone_map.csv")).set_index("station_id").weather_zone
    scored = archive.merge(history[[*KEY, "boardings"]], on=KEY, validate="many_to_one")
    scored = wonmi.for_prediction(scored, "baseline")
    evidence = [wonmi.compare(label, 3754, "2025-09-27", (10, 17), start, scored, protected, weather, zones)
                for label, start in (("summer", "2025-07-20"), ("2025-08-31", "2025-09-01"))]
    chosen = grid.loc[grid.date.eq("2026-10-17") & grid.station_id.eq(3754) & grid.hour.isin((13, 14, 15, 16))]
    shifts = {}
    for hour in chosen.hour:
        k = str(hour)
        cases = [part["dates"][part["event_date"]]["hourly"][k][2] for part in evidence]
        cases += [part["all_ordinary"]["neighbor_hourly_difference_in_differences"][k] for part in evidence]
        shifts[hour] = float(np.clip(.5 * 1700 / 2000 * min(cases), 0., .10))
    return add_offsets(grid, chosen.assign(log_delta=chosen.hour.map(shifts)), "wonmi_sports_day"), shifts


def later_festivals(grid, history, archive, protected, team, work):
    local = dream.episodes(archive, history.set_index(KEY).observed, protected)
    summary = dream.historical(local, 3207)
    selected = grid.loc[grid.date.isin(dream.FINAL_DAYS) & grid.station_id.eq(3207) & grid.hour.isin(dream.HOURS)].copy()
    shift = {}
    for part, case in summary.items():
        score = case["out_of_time_rmsle"]
        if score["candidate"] < score["reference"]:
            shift.update({(part, hour): value for hour, value in case["hourly_offsets"].items()})
    kind = np.where(selected.date.dt.day_of_week.ge(5), "weekend", "weekday")
    changes = [add_offsets(grid, selected.assign(log_delta=[shift.get(key, 0.) for key in zip(kind, selected.hour)]), "dream_park")]

    history_report = neunghudae.audit_history(team, work)
    selected = grid.date.isin(neunghudae.FORECAST_DATES) & grid.station_id.eq(3128) & grid.hour.eq(21)
    base = grid.loc[selected, "base"].to_numpy()
    levels = np.log1p(base) + np.minimum(np.log1p(neunghudae.CAP_COUNTS / (base + 1)), neunghudae.CAP_LOG)
    changes.append(update(grid, selected, levels, "neunghudae"))
    changes.extend(science(grid, history, work, refine=True))

    archive_report = trail.historical_check(archive, history.set_index(KEY).boardings)
    low = np.array([case["spatially_adjusted_contrast"] for case in archive_report.values()]).min(axis=0)
    shifts = dict(zip(trail.HOURS, np.clip(low / 3., 0., .08)))
    selected = grid.loc[grid.date.eq(trail.TARGET) & grid.station_id.eq(trail.STATION) & grid.hour.isin(trail.HOURS)]
    changes.append(add_offsets(grid, selected.assign(log_delta=selected.hour.map(shifts)), "trail_festival"))
    return changes, {"dream_park": summary, "neunghudae": history_report, "trail": archive_report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "baseline", "clean_oof", "output", "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--protected", type=Path)
    parser.add_argument("--early-protected", type=Path)
    parser.add_argument("--stage", choices=("before_model", "after_model"), required=True)
    args = parser.parse_args()
    team, work = args.team, args.work_dir
    grid, history, protected = inputs(team, work, args.baseline)
    archive = clean_history(args.clean_oof, history)
    if args.stage == "before_model":
        with TemporaryDirectory() as directory:
            changes = paired_removal(team, work, grid, Path(directory))
        changes.extend(historical_events(grid, history, protected, work))
        changes.extend(science(grid, history, work, refine=False))
        wonmi_rows, support = wonmi_adjustment(grid, history, archive, protected, work)
        changes.append(wonmi_rows)
        details = {"wonmi": support}
    else:
        changes, details = later_festivals(grid, history, archive, protected, team, work)
    edits = pd.concat(changes, ignore_index=True)
    if edits.example_id.duplicated().any():
        raise ValueError("Event rules in the same stage must have disjoint keys.")
    if args.protected:
        keys = edits[GRID]
        if args.stage == "before_model":
            keys = pd.concat([keys, initial_event_keys(grid)]).drop_duplicates(GRID)
        if args.early_protected:
            early = pd.read_csv(args.early_protected, parse_dates=["date"], usecols=GRID)
            keys = pd.concat([keys, early]).drop_duplicates(GRID)
        keys.to_csv(args.protected, index=False)
    save_submission(args.output, grid[["example_id"]], grid, grid.base.to_numpy())
    output = {"stage": args.stage, "input_sha256": digest(args.baseline), "output_sha256": digest(args.output),
              "clean_oof_sha256": digest(args.clean_oof), "rows": len(grid), "changed_rows": len(edits),
              "by_layer": edits.layer.value_counts().to_dict(), "details": details,
              "changes": json.loads(edits.to_json(orient="records", date_format="iso", double_precision=15))}
    args.report.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(output["by_layer"], output["output_sha256"], flush=True)


if __name__ == "__main__":
    main()
