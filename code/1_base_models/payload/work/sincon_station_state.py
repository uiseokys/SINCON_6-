"""Test whether a latent station level transfers across disjoint forecast periods."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.linalg import solveh_banded


KEY = ["date", "station_id", "hour"]
WINDOWS = {
    "A": ("2026-02-28", "2026-06-30"),
    "B": ("2025-12-31", "2026-04-30"),
    "C": ("2025-10-31", "2026-02-28"),
}
REGULAR = ["is_offday", "is_bridge_day", "is_quasi_holiday", "is_pre_long_offday", "is_post_long_offday"]
EVENT_STATIONS = [3127, 3139, 3209, 3754]


def read_team(team_dir, external_zip):
    train = pd.read_csv(team_dir / "data/train.csv", usecols=KEY + ["boardings"], parse_dates=["date"])
    meta = pd.read_csv(team_dir / "data/metaData.csv", usecols=["station_id", "canonical_line", "first_date"],
                       parse_dates=["first_date"]).set_index("station_id")
    predictions = pd.read_csv(team_dir / "valid_predictions_ABC.csv.gz", parse_dates=["date", "train_end"])
    with ZipFile(external_zip) as archive:
        calendar = pd.read_csv(archive.open("calendar_features.csv"), parse_dates=["date"]).set_index("date")
    if train.duplicated(KEY).any() or predictions.duplicated(["fold", *KEY]).any():
        raise ValueError("Expected a unique observation and one prediction per fold and key.")
    truth = train.set_index(KEY)["boardings"].reindex(pd.MultiIndex.from_frame(predictions[KEY]))
    actual = np.log1p(truth.to_numpy(dtype="float64"))
    if not np.allclose(actual, predictions["y_log"], atol=5.1e-7, rtol=0):
        raise ValueError("Archived validation predictions do not match the supplied training observations.")
    if not np.allclose((predictions["lgb_log"] + predictions["nn_log"]) / 2, predictions["ens_log"], atol=1.01e-6, rtol=0):
        raise ValueError("Archived ensemble differs from the team's documented log average.")
    if set(predictions["fold"]) != set(WINDOWS):
        raise ValueError("The archive must contain each documented fold.")
    for name, (cutoff, end) in WINDOWS.items():
        fold = predictions.loc[predictions["fold"].eq(name)]
        days = pd.date_range(pd.Timestamp(cutoff) + pd.Timedelta(days=1), end)
        if set(fold["train_end"]) != {pd.Timestamp(cutoff)} or set(fold["date"]) != set(days) or len(fold) != len(days) * 71 * 18:
            raise ValueError(f"Validation rows do not match the documented window {name}.")
    return train, meta, calendar, predictions.assign(y_log=actual)


def basis(dates, seasonal):
    index = pd.DatetimeIndex(dates)
    weekdays = np.column_stack([index.dayofweek == day for day in range(1, 5)])
    if not seasonal:
        return weekdays.astype(float)
    annual = 2 * np.pi * index.dayofyear.to_numpy() / 365.2425
    return np.column_stack([weekdays, np.sin(annual), np.cos(annual), np.sin(2 * annual), np.cos(2 * annual)])


def estimate_state(observed, future_dates, seasonal):
    dates = observed.index
    y = observed.to_numpy(dtype="float64")
    design = basis(dates, seasonal)
    roughness = 520 / np.diff(dates.to_numpy(dtype="datetime64[D]").astype(int))
    diag = 1 + np.r_[0, roughness] + np.r_[roughness, 0]
    banded = np.vstack([np.r_[0, -roughness], diag])
    filtered = solveh_banded(banded, np.column_stack([y, design]))
    filtered_y, filtered_x = filtered[:, 0], filtered[:, 1:]
    penalty = [2.] * 4 + ([10., 10., 24., 24.] if seasonal else [])
    beta = np.linalg.solve(design.T @ (design - filtered_x) + np.diag(penalty),
                           design.T @ (y - filtered_y))
    level = filtered_y[-1] - filtered_x[-1] @ beta
    return level + basis(future_dates, seasonal) @ beta


def relative_levels(frame, target_col, meta, calendar):
    ordinary = calendar.index[calendar[REGULAR].eq(0).all(axis=1)]
    workday = frame.loc[frame["date"].isin(ordinary) & frame["date"].dt.dayofweek.lt(5)]
    panel = workday.groupby(["date", "station_id"])[target_col].mean().unstack()
    centered = panel.copy()
    mature = meta["first_date"].eq(meta["first_date"].min()) & ~meta.index.isin(EVENT_STATIONS)
    for line, stations in meta.groupby("canonical_line").groups.items():
        keys = panel.columns.intersection(stations)
        peers = keys.intersection(meta.index[mature & meta["canonical_line"].eq(line)])
        values = panel[peers]
        peer_sum = values.sum(axis=1)
        peer_n = values.count(axis=1)
        for station in keys:
            self_values = values[station].fillna(0) if station in peers else 0
            own_count = values[station].notna().astype(int) if station in peers else 0
            centered[station] = panel[station] - (peer_sum - self_values) / (peer_n - own_count)
    return centered


def state_corrections(history, target, meta, calendar):
    if history["date"].max() >= target["date"].min() or {"boardings", "y_log"} & set(target):
        raise ValueError("State estimation requires history before the target and unlabeled predictions.")
    history = history.assign(log_boardings=np.log1p(history["boardings"].to_numpy(dtype="float64")))
    actual = relative_levels(history, "log_boardings", meta, calendar)
    reference = relative_levels(target, "ens_log", meta, calendar)
    output = {}
    for seasonal in (False, True):
        shifts = {}
        for station in reference:
            mature = meta.at[station, "first_date"] + pd.Timedelta(days=14)
            available = actual[station].dropna().loc[mature:] if station in actual else pd.Series(dtype=float)
            if len(available) < 25:
                continue
            estimate = estimate_state(available, reference.index, seasonal)
            shifts[station] = float(np.median(estimate - reference[station]))
        adjustment = pd.Series(shifts).reindex(reference.columns).fillna(0).clip(-0.3, 0.3)
        output["state_annual" if seasonal else "state_level"] = adjustment - adjustment.mean()
    return output


def preceding_errors(predictions, cutoff):
    earlier = predictions.loc[predictions["date"].le(cutoff)].copy()
    if earlier.empty:
        return earlier
    # Keep the oldest origin for each date to avoid giving overlapping dates two votes.
    origin = earlier.groupby("date")["train_end"].transform("min")
    earlier = earlier.loc[earlier["train_end"].eq(origin)]
    return earlier.assign(error=earlier["y_log"] - earlier["ens_log"])


def bias_corrections(earlier, meta):
    mean = earlier.groupby("station_id")["error"].mean()
    centered = mean - mean.mean()
    groups = meta["canonical_line"].astype(str) + "_" + meta["first_date"].eq(meta["first_date"].min()).astype(str)
    peer = centered.groupby(groups).transform("median")
    # Adjacent days share shocks, so use weekly means to estimate uncertainty.
    weekly = earlier.assign(week=earlier["date"].dt.to_period("W")).groupby(["station_id", "week"])["error"].mean()
    noise = weekly.groupby(level=0).var() / weekly.groupby(level=0).size()
    variation = (centered - peer).groupby(groups).transform("var")
    signal = (variation - noise.groupby(groups).transform("median")).clip(lower=0.0002)
    shrunken = peer + signal / (signal + noise) * (centered - peer)
    shrunken = shrunken.fillna(centered)
    return centered, shrunken - shrunken.mean()


def score_fold(train, meta, calendar, predictions, fold):
    cutoff = pd.Timestamp(WINDOWS[fold][0])
    historical = train.loc[train["date"].le(cutoff)]
    observed = predictions.loc[predictions["fold"].eq(fold)]
    target = observed[KEY + ["ens_log"]]
    shifts = state_corrections(historical, target, meta, calendar)
    earlier = preceding_errors(predictions, cutoff)
    if len(earlier):
        raw, pooled = bias_corrections(earlier, meta)
        shifts.update(previous_bias=raw, pooled_bias=pooled, pooled_bias_with_state=0.5 * pooled + 0.5 * shifts["state_annual"])
    additions = {"ensemble": np.zeros(len(target))} | {
        name: target["station_id"].map(shift).fillna(0).to_numpy() for name, shift in shifts.items()
    }
    actual = observed["y_log"].to_numpy()
    base = target["ens_log"].to_numpy()
    return {"calibration_days": int(earlier["date"].nunique()),
            "rmsle": {name: float(np.sqrt(np.mean((actual - np.maximum(base + change, 0)) ** 2)))
                      for name, change in additions.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team_dir", type=Path)
    parser.add_argument("external_zip", type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    team = read_team(args.team_dir, args.external_zip)
    report = {fold: score_fold(*team, fold) for fold in WINDOWS}
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.results:
        args.results.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
