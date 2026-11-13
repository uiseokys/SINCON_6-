"""Create uncalibrated TabICL predictions from official historical observations."""

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from tabicl import TabICLRegressor

from sincon_joint_candidate import setup
from sincon_station_state import WINDOWS
from sincon_tabicl_experiment import forecast, station_days


CHECKPOINT_SHA256 = "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a"
WINDOWS = {**WINDOWS, "summer": ("2025-06-30", "2025-10-31"),
           "2025-07-31": ("2025-07-31", "2025-11-30"), "2025-08-31": ("2025-08-31", "2025-12-31"),
           "2025-09-30": ("2025-09-30", "2026-01-31"), "2026-03-31": ("2026-03-31", "2026-06-30"),
           "2026-04-30": ("2026-04-30", "2026-06-30"), "final": ("2026-06-30", "2026-10-31")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "external", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--folds", choices=WINDOWS, nargs="+", required=True)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Expected the original TabICL v2 checkpoint.")
    work = args.work_dir
    data, _, _, _, stations, _, _ = setup(args.team, args.external, work / "sincon_kbo_game_data.csv",
                                          work / "mobility_monthly", work / "sincon_station_addresses.csv", args.threads)
    observed = data.loc[data.boardings.notna(), "date"]
    if len(observed) != 688_176 or observed.max() != pd.Timestamp("2026-06-30"):
        raise ValueError("Only the supplied historical labels may enter TabICL.")
    table, daily = station_days(data)
    model = TabICLRegressor(model_path=args.checkpoint, allow_auto_download=False, n_estimators=2,
                            device="cpu", n_jobs=args.threads, random_state=2026)
    results = []
    for label in args.folds:
        cutoff, end = map(pd.Timestamp, WINDOWS[label])
        part = forecast(table, daily, cutoff, end, stations.index, model, 1).drop(columns="band")
        expected = (end - cutoff).days * len(stations)
        if len(part) != expected or part.duplicated(["date", "station_id"]).any() or not part.date.gt(cutoff).all():
            raise ValueError(f"Incomplete station-day predictions for {label}.")
        results.append(part.assign(label=label, origin=cutoff))
        print(f"{label} {len(part)} station-days", flush=True)
    pd.concat(results, ignore_index=True).to_csv(args.output, index=False, float_format="%.17g")


if __name__ == "__main__":
    main()
