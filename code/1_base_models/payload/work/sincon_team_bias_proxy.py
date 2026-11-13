"""Compare source weights under a chronological proxy for the released team station correction."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_apply_final_calibration import PUBLIC_STATION_WEIGHT
from sincon_source_reweight import FIELDS, evaluate
from sincon_station_state import KEY, WINDOWS


def earlier_team_errors(archive, cutoff):
    window = archive.date.gt(cutoff - pd.Timedelta(days=120)) & archive.date.le(cutoff)
    usable = archive.loc[window & archive.origin.lt(cutoff) & archive.station_age.ge(60)]
    usable = usable.sort_values("origin").drop_duplicates(KEY, keep="last")
    means = (usable.y_log - usable.team).groupby(usable.station_id).mean()
    return PUBLIC_STATION_WEIGHT * (means - means.mean()), usable.date.nunique()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team", type=Path)
    parser.add_argument("earlier", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    work = args.work_dir
    columns = [*KEY, "team", "y_log"]
    dates = {fold: pd.Timestamp(period[0]) for fold, period in WINDOWS.items()}
    current = pd.read_csv(work / "sincon_local_validation.csv.gz", usecols=[*columns, "fold"], parse_dates=["date"])
    summer = pd.read_csv(work / "sincon_summer_replay_validation.csv.gz", usecols=columns, parse_dates=["date"])
    first_seen = pd.read_csv(args.team / "data/metaData.csv", parse_dates=["first_date"]).set_index("station_id").first_date
    archive = pd.concat([current.assign(origin=current.fold.map(dates)),
                         summer.assign(origin=pd.Timestamp("2025-06-30"))], ignore_index=True)
    archive["station_age"] = (archive.origin - archive.station_id.map(first_seen)).dt.days
    if archive.origin.ge(archive.date).any() or archive.station_age.isna().any():
        raise ValueError("Invalid origin or station metadata in archived forecasts.")
    weights, rows = evaluate(work, args.team, args.earlier)
    report = {}
    for fold, cutoff in dates.items():
        target = rows.loc[rows.fold.eq(fold)]
        station_shift, days = earlier_team_errors(archive, cutoff)
        if days < 60:
            raise ValueError(f"Insufficient pre-origin observations for fold {fold}.")
        shifts = {"local_stack_proxy": target.station_shift,
                  "team_forecast_proxy": target.station_id.map(station_shift).fillna(target.station_shift)}
        weights_by_source = [weights["folds"][fold]["weights"][source] for source in FIELDS]
        corrected = target.local_stack + target[FIELDS].to_numpy() @ weights_by_source
        actual = target.y_log.to_numpy()
        scores = {}
        for name, shift in shifts.items():
            forecasts = np.column_stack([target.local_stack + shift, corrected + shift])
            a, b = np.sqrt(np.square(actual[:, None] - np.maximum(forecasts, 0)).mean(axis=0))
            scores[name] = {"existing": float(a), "reweighted": float(b), "improvement": float(a - b)}
        report[fold] = {"rows": len(target), "prior_days": days, "stations_with_team_errors": len(station_shift), **scores}
    output = work / "sincon_team_bias_proxy_results.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
