"""Test zone weather and local non-subway mobility inside the team's tree model."""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sincon_daily_network import prepare_data
from sincon_district_mobility import CODE_CHANGES, station_districts
from sincon_mobility_data import MODES
from sincon_station_state import KEY, WINDOWS


def attach_mobility(frame, directory, station_district):
    raw = pd.concat((pd.read_csv(path, dtype={"district": str}, parse_dates=["date"])
                     for path in sorted(directory.glob("*.csv.gz"))), ignore_index=True)
    if set(raw["mode"]) != set(MODES.values()):
        raise ValueError("Expected the four non-subway modes in the aggregate input.")
    raw["district"] = raw["district"].replace(CODE_CHANGES)
    modes = raw.groupby(["date", "hour", "district", "mode"])["arrivals"].sum().unstack("mode", fill_value=0)
    total = modes.groupby(["date", "district"]).transform("sum")
    x = np.log1p(modes).add_prefix("arrivals_")
    shares = (np.log1p(modes) - np.log1p(total)).add_prefix("hourly_share_")
    total = np.log1p(total).add_prefix("daily_arrivals_")
    mobility = pd.concat([x, shares, total], axis=1)
    frame = frame.assign(district=frame["station_id"].map(station_district))
    enriched = frame.join(mobility, on=["date", "hour", "district"], validate="many_to_one")
    if enriched[mobility.columns].isna().any().any():
        raise ValueError("Mobility data must cover all station-hours.")
    return enriched, list(mobility.columns)


def run(team_dir, external_zip, scores, directory, address_csv, threads, seeds, output):
    frame, validation, _ = prepare_data(team_dir, external_zip, scores)
    from ensemble import FEATURES
    from improved import PARAMS, fit_predict

    PARAMS["num_threads"] = threads
    frame, mobility_features = attach_mobility(frame, directory, station_districts(address_csv, team_dir))
    models = {
        "zone_weather": FEATURES,
        "zone_weather_and_finish": FEATURES + ["mh_hours_from_end", "mh_audience_here"],
        "zone_weather_and_mobility": FEATURES + mobility_features,
    }
    report = {"folds": {}, "lgb_seeds": list(seeds)}
    for name, (cutoff, end) in WINDOWS.items():
        past = frame.loc[frame["date"].le(cutoff)]
        future = frame.loc[frame["date"].gt(cutoff) & frame["date"].le(end)].drop(columns="boardings")
        archived = validation.loc[validation["fold"].eq(name)].set_index(KEY).reindex(pd.MultiIndex.from_frame(future[KEY]))
        actual = archived["y_log"].to_numpy()
        neural = archived["nn_log"].to_numpy()
        if not np.isfinite(np.column_stack([actual, neural])).all():
            raise ValueError(f"Incomplete archived prediction coverage in {name}.")
        result = {"ensemble": float(np.sqrt(np.mean(np.square(archived["ens_log"] - actual))))}
        for key, features in models.items():
            started = perf_counter()
            log_lgb = np.log1p(fit_predict(past, future, features, seeds))
            log_ensemble = (log_lgb + neural) / 2
            result[key] = {"alone": float(np.sqrt(np.mean(np.square(log_lgb - actual)))),
                           "ensemble": float(np.sqrt(np.mean(np.square(log_ensemble - actual)))),
                           "seconds": round(perf_counter() - started, 2)}
            report["folds"][name] = result
            if output:
                output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(json.dumps({name: {key: result[key]}}, allow_nan=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team_dir", type=Path)
    parser.add_argument("external_zip", type=Path)
    parser.add_argument("scores", type=Path)
    parser.add_argument("mobility_directory", type=Path)
    parser.add_argument("station_addresses", type=Path)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    run(args.team_dir, args.external_zip, args.scores, args.mobility_directory, args.station_addresses,
        args.threads, range(args.seeds), args.results)
