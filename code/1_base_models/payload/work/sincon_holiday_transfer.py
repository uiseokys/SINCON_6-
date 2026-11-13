"""Transfer previous occurrences of named holidays to fixed-origin forecasts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_joint_candidate import rmsle_logs
from sincon_station_state import KEY, WINDOWS, read_team


EXCLUSIONS = ["is_holiday", "is_bridge_day", "is_quasi_holiday", "is_lunar_season", "is_pre_long_offday", "is_post_long_offday"]


def signature(calendar):
    signatures = {}
    names = calendar["holiday_name"].dropna()
    for day, name in names.items():
        signatures[day] = [(name, 1.)]
        if "·" in name and not name.startswith("대체공휴일"):
            signatures[day].extend((part, .55) for part in name.split("·"))
        if name.startswith("대체공휴일(") and name.endswith(")"):
            core = name[6:-1]
            signatures[day].append((core, .30))
    for day, name in names.items():
        if name not in ("설날", "추석"):
            continue
        for offset in range(-2, 3):
            nearby = day + pd.Timedelta(days=offset)
            if nearby in calendar.index and calendar.at[nearby, "is_offday"]:
                signatures.setdefault(nearby, []).insert(0, (f"{name} {offset:+d}", 1.))
    return signatures


def ordinary_profile(panel, date, ordinary, fallback=None):
    distance = np.abs((panel.index - date).days)
    rows = (distance <= 65) & panel.index.isin(ordinary)
    subset = panel.loc[rows]
    if len(subset) >= 4:
        return subset.median(axis=0)
    if fallback is not None:
        return fallback
    return panel.loc[panel.index.isin(ordinary)].median(axis=0)


def transfer(history, future, calendar, meta, source, cutoff):
    if history["date"].max() > cutoff or future["date"].min() <= cutoff or {"boardings", "y_log"} & set(future):
        raise ValueError("Named-holiday profiles require pre-origin ridership and unlabeled future rows.")
    forecasts = future.pivot(index="date", columns=["station_id", "hour"], values=source)
    past = history.assign(log=np.log1p(history["boardings"])).pivot(index="date", columns=["station_id", "hour"], values="log")
    regular = calendar.index[(calendar.index.dayofweek == 6) & calendar[EXCLUSIONS].eq(0).all(axis=1)]
    signatures = signature(calendar)
    results = {name: forecasts.copy() for name in ("same_holiday", "pooled_same_holiday")}
    eligible = []
    for date in forecasts.index.intersection(signatures):
        possible = dict(signatures[date])
        donors = {}
        for donor in past.index.intersection(signatures):
            similarity = max((strength * possible.get(name, 0) for name, strength in signatures[donor]), default=0)
            if similarity > 0:
                donors[donor] = similarity
        if not donors:
            continue
        # Prefer the same named day over substitutions or a joint celebration.
        best = max(donors.values())
        selected = [donor for donor, weight in donors.items() if weight >= best - .05]
        profiles = pd.DataFrame([past.loc[day] - ordinary_profile(past, day, regular) for day in selected])
        effect = profiles.median().reindex(forecasts.columns).clip(-1.6, 1.6)
        line_hour = [effect.index.get_level_values("station_id").map(meta["canonical_line"]), effect.index.get_level_values("hour")]
        pooled = effect.groupby(line_hour).transform("median")
        equivalent = ordinary_profile(forecasts, date, regular)
        baseline = forecasts.loc[date]
        blend = .55 if best >= .9 else .25
        for name, local in [("same_holiday", effect), ("pooled_same_holiday", 0.8 * effect + 0.2 * pooled)]:
            difference = (equivalent + local - baseline).clip(-0.7, .7)
            results[name].loc[date] = baseline + blend * difference.fillna(0)
        eligible.append(str(date.date()))
    index = pd.MultiIndex.from_frame(future[KEY])
    return {name: frame.stack(future_stack=True).stack(future_stack=True).reorder_levels(KEY).reindex(index).to_numpy()
            for name, frame in results.items()}, eligible


def run(args):
    train, meta, calendar, _ = read_team(args.team_dir, args.external_zip)
    validation = pd.read_csv(args.validation, parse_dates=["date"])
    results, outputs = {}, []
    for fold, (origin, _) in WINDOWS.items():
        cutoff = pd.Timestamp(origin)
        past = train.loc[train["date"].le(cutoff)]
        actual = validation.loc[validation["fold"].eq(fold)]
        target = actual[KEY + ["local_stack"]]
        estimates, dates = transfer(past, target, calendar, meta, "local_stack", cutoff)
        variants = {"reference": target["local_stack"].to_numpy(), **estimates}
        relevant = target["date"].isin(pd.to_datetime(dates)).to_numpy()
        truth = actual["y_log"].to_numpy()
        if not np.isfinite(np.column_stack([truth, *variants.values()])).all():
            raise ValueError(f"Incomplete holiday forecasts in {fold}.")
        results[fold] = {"rmsle": {name: rmsle_logs(truth, values) for name, values in variants.items()},
                         "holiday_rmsle": {name: rmsle_logs(truth[relevant], values[relevant]) for name, values in variants.items()},
                         "adjusted_dates": dates}
        outputs.append(target[KEY].assign(fold=fold, y_log=truth, **variants))
        print(json.dumps({fold: results[fold]}, ensure_ascii=False), flush=True)
    args.results.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if args.predictions:
        pd.concat(outputs, ignore_index=True).to_csv(args.predictions, index=False, float_format="%.9f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team_dir", "external_zip", "validation"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    run(parser.parse_args())
