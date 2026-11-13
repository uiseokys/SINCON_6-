"""Audit the recurring Incheon Grand Park trail festival using historical forecasts."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


KEY = ["date", "station_id", "hour"]
STATION = 3226
NEARBY = [3224, 3225, 3227]
HOURS = [12, 13]
DONOR = pd.Timestamp("2025-10-19")
TARGET = pd.Timestamp("2026-10-31")
CONTROL_DATES = pd.to_datetime(["2025-10-26", "2025-11-02", "2025-11-09", "2025-11-16"])
ORIGINS = {"summer": "2025-06-30", "2025-07-31": "2025-07-31", "2025-08-31": "2025-08-31", "2025-09-30": "2025-09-30"}
BASE_SHA256 = "2d836ed13e1a47be96575d16e5abe843c301407e3ef6302422ae049e837741da"
SOURCES = {
    "historical": "https://www.incheon.go.kr/mayor_8/MA040101/view?repSeq=DOM_0000000013314801",
    "target": "https://www.incheon.go.kr/IC010205/view?repSeq=DOM_0000000015868742",
    "target_schedule": "https://www.icbp.go.kr/dong/bbs/bbsMsgDetail.do?bcd=dong_notice&dong=cheongcheon2&msg_seq=136242&pgno=1",
    "exclude_2025_10_12_art_festival": "https://www.incheon.go.kr/IC010101/view?nttNo=2045250",
}


def historical_check(replay, truth):
    date_scope = CONTROL_DATES.insert(0, DONOR)
    report = {}
    for label, origin in ORIGINS.items():
        frame = replay.loc[replay.label.eq(label) & replay.date.isin(date_scope)
                           & replay.station_id.isin([STATION, *NEARBY]) & replay.hour.isin(HOURS)].copy()
        if (not frame.date.gt(pd.Timestamp(origin)).all() or frame.date.max() > truth.index.get_level_values("date").max()
                or frame.duplicated(KEY).any()):
            raise ValueError(f"Unexpected historical replay coverage for {label}.")
        observed = truth.reindex(pd.MultiIndex.from_frame(frame[KEY]))
        if not np.allclose(frame.y_log, np.log1p(observed), rtol=0., atol=1e-9):
            raise ValueError(f"Held-out labels differ from training archive for {label}.")
        frame["residual"] = frame.y_log - frame.baseline
        residual = frame.pivot(index=["date", "hour"], columns="station_id", values="residual")
        adjusted = (residual[STATION] - residual[NEARBY].median(axis=1)).unstack("hour")
        controls = adjusted.drop(DONOR)
        contrast = adjusted.loc[DONOR] - controls.median(axis=0)
        source = frame.loc[frame.station_id.eq(STATION) & frame.date.eq(DONOR)].sort_values("hour")
        report[label] = {
            "forecast_origin": origin, "hours": HOURS,
            "observed": np.rint(np.expm1(source.y_log)).astype(int).tolist(),
            "forecast": np.expm1(source.baseline).tolist(),
            "control_dates": [date.date().isoformat() for date in controls.index],
            "spatially_adjusted_contrast": contrast.tolist(),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "archive", "baseline", "weather_bundle", "report", "delta"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    truth = pd.read_csv(args.team / "data/train.csv", parse_dates=["date"], usecols=[*KEY, "boardings"])
    if truth.date.max() != pd.Timestamp("2026-06-30"):
        raise ValueError("Use the competition's training-period observations.")
    report = historical_check(pd.read_pickle(args.archive), truth.set_index(KEY).boardings)
    with ZipFile(args.weather_bundle) as archive:
        root = "external_inputs/external/"
        zones = pd.read_csv(archive.open(root + "weather_zone_map.csv")).set_index("station_id").weather_zone
        weather = pd.read_csv(archive.open(root + "weather_hourly.csv"), parse_dates=["date"],
                              usecols=["date", "weather_zone", "hour", "temp", "precip"])
    weather = weather.loc[weather.weather_zone.eq(zones[STATION]) & weather.date.isin([DONOR, TARGET]) & weather.hour.between(10, 15)]
    weather_stats = weather.groupby("date").agg(temp_mean_c=("temp", "mean"), rain_mm=("precip", "sum"))
    grid = pd.read_csv(args.team / "data/test.csv", parse_dates=["date"])
    baseline = pd.read_csv(args.baseline, float_precision="round_trip")
    if (hashlib.sha256(args.baseline.read_bytes()).hexdigest() != BASE_SHA256 or len(grid) != 157_194
            or not grid.example_id.equals(baseline.example_id)):
        raise ValueError("Expected the verified, full-grid model update.")
    rows = grid.loc[grid.date.eq(TARGET) & grid.station_id.eq(STATION) & grid.hour.isin(HOURS)].sort_values("hour").copy()
    if rows.example_id.tolist() != ["J156966", "J156967"]:
        raise ValueError("The adjustment scope must match both documented station-hours.")
    minimum = np.array([case["spatially_adjusted_contrast"] for case in report.values()]).min(axis=0)
    if (minimum <= 0.).any():
        raise ValueError("Both proposed hours need consistent historical support.")
    rows["log_delta"] = np.minimum(minimum / 3., .08)
    rows[["example_id", *KEY, "log_delta"]].to_csv(args.delta, index=False, float_format="%.12f")
    before = baseline.set_index("example_id").boardings.reindex(rows.example_id).to_numpy()
    after = np.expm1(np.log1p(before) + rows.log_delta.to_numpy())
    result = {
        "sources": SOURCES, "historical_replays": report,
        "weather": json.loads(weather_stats.reset_index().to_json(orient="records", date_format="iso")),
        "scope": "Single historical edition. 2025 used Incheon trail course 6. 2026 used the park and Gwanmosan route.",
        "delta_sha256": hashlib.sha256(args.delta.read_bytes()).hexdigest(),
        "rows": json.loads(rows.assign(before=before, after=after).to_json(orient="records", date_format="iso", double_precision=15)),
    }
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
