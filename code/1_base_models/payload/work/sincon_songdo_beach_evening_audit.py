"""Audit the 19–20h residual from the incorrect 3139 Songdo Beach event flag."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sincon_songdo_beach_location_lgb import FINAL_DAYS, HISTORICAL_DAYS, KEY, SOURCES, WRONG_STATION
from sincon_vector_shape_final import CANCELLATION_SHA256
from sincon_weekly_variability import protected_days


HOURS = (19, 20)
TRANSFER = .25
TARGET_NAME = "submission_0p15126_event_cancel_fix.csv"
HISTORICAL_SOURCES = {
    "host_2025_dates_and_venue": "https://www.yeonsu.go.kr/main/community/notify/report.asp?page=v&seq=28076",
    "host_2025_august13_suspension": "https://www.yeonsu.go.kr/main/community/notify/notice.asp?page=v&seq=114222",
}


def sum_readout(rows):
    return {str(hour): dict(zip(("rows", "observed_or_prior_sum", "archive_or_adjusted_sum"),
                                [len(part), float(part.observed.sum()), float(part.archive.sum())]))
            for hour, part in rows.groupby("hour")}


def historical_scores(truth, prior, effect, dates):
    old = np.square(truth - prior.clip(min=0.))
    new = np.square(truth - (prior + effect).clip(min=0.))
    before, after = float(np.sqrt(old.mean())), float(np.sqrt(new.mean()))
    leave_one = {}
    for date in dates.unique():
        keep = dates.ne(date)
        leave_one[str(date.date())] = float(np.sqrt(old[keep].mean()) - np.sqrt(new[keep].mean()))
    return {"reference": before, "candidate": after, "leave_one_day_out_gain": leave_one}


def paired(work, grid, team):
    with ZipFile(work / "sincon_songdo_beach_location_responses.zip") as bundle:
        report = json.load(bundle.open("sincon_songdo_beach_location_lgb_results.json"))
        tree = pd.DataFrame(report["final_changed_scope"])
        neural = [pd.read_csv(bundle.open(f"sincon_songdo_beach_nn_seed{i}.csv")) for i in range(3)]
    exact = pd.read_csv(team / "ensemble_parts.csv").set_index("example_id")
    target = grid.loc[grid.station_id.eq(WRONG_STATION) & grid.date.isin(FINAL_DAYS) & grid.hour.isin(HOURS)].copy()
    if len(target) != 16:
        raise ValueError("Expected two hours on every one of the eight dates.")
    target = target.set_index("example_id", drop=False)
    archived = exact.reindex(target.index)
    tree = tree.set_index("example_id").reindex(target.index)
    neural = [part.set_index("example_id").reindex(target.index) for part in neural]
    if not tree[KEY].equals(target[KEY]) or not np.allclose(tree.original_log_lgb, archived.log_lgb, atol=1e-8, rtol=0.):
        raise ValueError("The scenario must match original tree predictions and the official test keys.")
    for part in neural:
        if (not part[KEY].equals(target[KEY]) or not np.allclose(part.archived_nn, archived.log_nn, atol=1e-8, rtol=0.)
                or not np.array_equal(part.station_delta, part.combined_delta)):
            raise ValueError("A relocation changes the station exposure while retaining the city event count.")
    responses = np.column_stack([tree.delta_log, *[part.station_delta for part in neural]])
    if not np.isfinite(responses).all() or np.any(responses >= 0.):
        raise ValueError("All paired responses must lower the falsely flagged 19–20h rows.")
    return target.reset_index(drop=True).assign(lgb=responses[:, 0], neural=responses[:, 1:].mean(axis=1),
                                                paired=.5 * (responses[:, 0] + responses[:, 1:].mean(axis=1)))


def audit(args):
    team, work = args.team, args.work_dir
    baseline = team / TARGET_NAME
    if hashlib.sha256(baseline.read_bytes()).hexdigest() != CANCELLATION_SHA256:
        raise ValueError("This comparison requires the user-scored 0.14774 incumbent.")
    grid = pd.read_csv(team / "data/test.csv", usecols=[*KEY, "example_id"])
    incumbent = pd.read_csv(baseline)
    if len(grid) != 157_194 or not grid.example_id.equals(incumbent.example_id) or grid.duplicated(KEY).any():
        raise ValueError("Expected the original 157,194 unlabeled test keys.")
    target = paired(work, grid, team).merge(incumbent, on="example_id", validate="one_to_one")
    target["log_delta"] = TRANSFER * target.paired
    target["counterfactual"] = np.expm1(np.log1p(target.boardings) + target.log_delta)
    target["full_flag_removal"] = np.expm1(np.log1p(target.boardings) + target.paired)

    official = pd.read_csv(team / "data/train.csv", usecols=[*KEY, "boardings"], parse_dates=["date"])
    archive = pd.read_csv(args.rolling, usecols=[*KEY, "origin", "y_log", "local_stack"], parse_dates=["date", "origin"])
    historical_dates = pd.DatetimeIndex(HISTORICAL_DAYS)
    keep = archive.station_id.eq(WRONG_STATION) & archive.hour.isin(HOURS) & archive.date.isin(historical_dates)
    old = archive.loc[keep].merge(official, on=KEY, validate="one_to_one")
    if (len(old) != 14 or not old.origin.eq("2025-07-31").all() or not old.date.gt(old.origin).all()
            or not np.allclose(old.y_log, np.log1p(old.boardings), atol=1e-8, rtol=0.)):
        raise ValueError("Historical evidence requires all 14 official held-out records.")
    mapping = target.assign(date=pd.to_datetime(target.date) - pd.Timedelta(days=364))
    analog = old.merge(mapping[[*KEY, "paired", "log_delta"]], on=KEY, validate="one_to_one")
    if len(analog) != len(old):
        raise ValueError("Each held-out historical hour needs a same-weekday paired response.")
    truth, prior, response, chosen = (analog[column].to_numpy() for column in ("y_log", "local_stack", "paired", "log_delta"))
    historical = {}
    for name, effect in (("quarter_transfer", chosen), ("half_sensitivity", .5 * response), ("full_sensitivity", response)):
        historical[name] = {scope: historical_scores(truth[use], prior[use], effect[use], analog.date[use])
                            for scope, use in (("all", np.ones(len(analog), dtype=bool)),
                                               ("19", analog.hour.eq(19)), ("20", analog.hour.eq(20)))}
    normals = official.loc[official.station_id.eq(WRONG_STATION) & official.hour.isin(HOURS)
                           & official.date.between("2026-05-01", "2026-06-30")].copy()
    holidays, events = protected_days(work)
    normals = normals.loc[~normals.date.isin(holidays) & ~pd.MultiIndex.from_frame(normals[KEY[:2]]).isin(events)]
    anchor = normals.assign(weekday=normals.date.dt.day_of_week).groupby(["weekday", "hour"]).boardings.agg(["median", "size"])
    target["normal_2026"] = anchor["median"].reindex(pd.MultiIndex.from_arrays([pd.to_datetime(target.date).dt.day_of_week,
                                                                              target.hour])).to_numpy()
    if target.normal_2026.isna().any():
        raise ValueError("Every proposed adjustment needs a pre-cutoff weekday anchor.")
    latest = target[["example_id", *KEY, "boardings", "lgb", "neural", "log_delta", "counterfactual", "full_flag_removal", "normal_2026"]]
    return {"sources": {**SOURCES, **HISTORICAL_SOURCES}, "origin": "2025-07-31", "final_cutoff": "2026-06-30", "transfer": TRANSFER,
            "historical": historical,
            "historical_counts": sum_readout(analog.assign(observed=analog.boardings, archive=np.expm1(analog.local_stack))),
            "historical_overpredicted": int((analog.local_stack > analog.y_log).sum()),
            "unmatched_target_date": "2026-08-15",
            "target_hour_means": latest.groupby("hour")[["boardings", "counterfactual", "full_flag_removal", "normal_2026"]].mean().to_dict("index"),
            "keyed_increments": latest.to_dict("records")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "rolling", "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = audit(args)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("historical", "historical_counts", "target_hour_means")}, indent=2))


if __name__ == "__main__":
    main()
