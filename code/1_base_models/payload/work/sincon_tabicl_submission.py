"""Build the official submission from independently verified log adjustments."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_munhak_ordinary import WARM
from sincon_network_level_audit import ordinary
from sincon_songdo_schedule import save_submission
from sincon_station_state import KEY
from sincon_tabicl_final_delta import BASELINE, aligned_counts, digest, protect_previous_layers
from sincon_weather_sensitivity import score


def read_delta(path, grid, expected_hash, full):
    if digest(path) != expected_hash:
        raise ValueError(f"The adjustment file checksum differs for {path.name}.")
    table = pd.read_csv(path, parse_dates=["date"], float_precision="round_trip")
    columns = ["example_id", *KEY, "log_delta"]
    if list(table.columns) != columns or not table.example_id.is_unique or not np.isfinite(table.log_delta).all():
        raise ValueError(f"Invalid keyed log adjustments in {path.name}.")
    matched = grid.set_index("example_id").reindex(table.example_id)
    if not table[KEY].equals(matched[KEY].reset_index(drop=True)) or (full and not table.example_id.equals(grid.example_id)):
        raise ValueError(f"The adjustment keys differ from the official grid in {path.name}.")
    return table.set_index("example_id").log_delta.reindex(grid.example_id, fill_value=0.).to_numpy()


def paired_validation(args):
    keys = ["label", "origin", *KEY]
    rows = pd.read_pickle(args.validation_replay)
    tabicl = pd.read_pickle(args.tabicl_oof).rename(columns={"log_delta": "tabicl"})
    retail = pd.read_csv(args.retail_oof, parse_dates=["date", "origin"]).rename(columns={"log_delta": "retail"})
    known = pd.MultiIndex.from_frame(rows[keys])
    if any(not pd.MultiIndex.from_frame(frame[keys]).isin(known).all() for frame in (tabicl, retail)):
        raise ValueError("Every overlay must match the identical archived forecast keys.")
    rows = rows.merge(tabicl, on=keys, how="left", validate="one_to_one")
    rows = rows.merge(retail, on=keys, how="left", validate="one_to_one")
    rows["retail"] = rows.retail.fillna(0.)
    samples = {label: rows.loc[rows.label.eq(label)] for label in "ABC"}
    samples["disjoint_A_C"] = pd.concat([samples["A"], samples["C"]])
    samples["disjoint_warm"] = pd.concat([rows.loc[rows.label.eq(label) & rows.date.dt.month.eq(month)] for label, month in WARM])
    report = {}
    for label, sample in samples.items():
        if sample.tabicl.isna().any() or sample.duplicated(KEY).any():
            raise ValueError("Each validation sample needs complete overlay coverage and disjoint dates.")
        baseline, actual = sample.baseline.to_numpy(), sample.y_log.to_numpy()
        tabicl = baseline + sample.tabicl.to_numpy()
        combined = tabicl + sample.retail.to_numpy()
        report[label] = {"vs_level_neutral": score(actual, baseline, combined, sample.date),
                         "vs_tabicl": score(actual, tabicl, combined, sample.date),
                         "retail_rows": int(sample.retail.ne(0.).sum())}
    return report


def run(args):
    test_path = args.team / "data/test.csv"
    base_path = args.work_dir / BASELINE
    manifest = json.loads(args.tabicl_report.read_text())
    hashes = manifest["sha256"]
    if digest(test_path) != hashes["test"] or digest(base_path) != hashes["incumbent"]:
        raise ValueError("The adjustments were fitted on a different test grid or incumbent.")
    grid = pd.read_csv(test_path, parse_dates=["date"], usecols=["example_id", *KEY])
    if len(grid) != 157_194 or not grid.example_id.is_unique or grid.duplicated(KEY).any():
        raise ValueError("The submission needs all official test station-hours in order.")
    counts = aligned_counts(base_path, grid)
    protected_days, masks = protect_previous_layers(grid, counts, args.work_dir, args.team)
    protected = pd.DataFrame(masks).any(axis=1).to_numpy()
    tabicl = read_delta(args.tabicl, grid, hashes["delta"], full=True)
    if np.count_nonzero(tabicl) != manifest["changed_rows"] or np.any(tabicl[protected]):
        raise ValueError("The TabICL adjustment altered verified event rows or differs from the checked manifest.")
    adjustment = tabicl.copy()
    retail_rows = 0
    if args.retail:
        retail_manifest = json.loads(args.retail_report.read_text())
        if retail_manifest["incumbent"]["sha256"] != hashes["incumbent"]:
            raise ValueError("The Sunday adjustments were fitted for a different incumbent.")
        retail_hash = retail_manifest["artifacts"][args.retail.name]["sha256"]
        retail = read_delta(args.retail, grid, retail_hash, full=False)
        affected = retail != 0.
        actual_days = sorted(grid.date[affected].dt.strftime("%Y-%m-%d").unique())
        expected_days = sorted(retail_manifest["final"]["closed"] + retail_manifest["final"]["open"])
        scope = ordinary(grid, protected_days) & grid.station_id.eq(3131) & grid.hour.between(10, 22) & grid.date.dt.day_of_week.eq(6)
        retail_rows = int(affected.sum())
        if (retail_rows != retail_manifest["final"]["rows"] or np.any(affected & ~scope)
                or actual_days != expected_days or not np.isclose(retail.sum(), 0., atol=1e-10)):
            raise ValueError("The Sunday adjustments must match the audited dates and preserved station-day scopes.")
        adjustment += retail
    values = counts.to_numpy(copy=True)
    changed = adjustment != 0.
    values[changed] = np.expm1(np.log1p(values[changed]) + adjustment[changed])
    if not np.array_equal(values[protected], counts.to_numpy()[protected]):
        raise ValueError("An adjustment altered a verified prior event prediction.")
    save_submission(args.submission, grid[["example_id"]].assign(boardings=counts), grid, values)
    return {"submission": str(args.submission.resolve()), "rows": len(values),
            "changed_rows": int(np.count_nonzero(values != counts.to_numpy())),
            "retail_rows": retail_rows,
            "protected_rows": int(protected.sum()), "min_boardings": float(values.min()), "max_boardings": float(values.max()),
            "sha256": hashlib.sha256(args.submission.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "tabicl", "tabicl_report", "submission", "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--retail", type=Path)
    parser.add_argument("--retail-report", type=Path)
    parser.add_argument("--validation-replay", type=Path)
    parser.add_argument("--tabicl-oof", type=Path)
    parser.add_argument("--retail-oof", type=Path)
    args = parser.parse_args()
    if bool(args.retail) != bool(args.retail_report):
        parser.error("Provide the Sunday adjustment and its audit together.")
    validation_args = (args.validation_replay, args.tabicl_oof, args.retail_oof)
    if any(validation_args) and not all(validation_args):
        parser.error("Provide all three validation archives together.")
    validation = paired_validation(args) if all(validation_args) else None
    result = run(args)
    if validation:
        result["validation"] = validation
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
