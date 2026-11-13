"""Check historical train-only analogs for two source-confirmed local events."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sincon_station_state import KEY
from sincon_tabicl_day_weight import BASE_WEIGHT, fit_weight, guarded_weight, load_replay
from sincon_weekly_variability import protected_days

NEARBY = {3754: (3753, 3755, 3756), 3226: (3224, 3225, 3227)}
LOCAL = sorted({*NEARBY, *(s for group in NEARBY.values() for s in group)})
SOURCES = {
    "wonmi_2025": "https://news.bucheon.go.kr/home/kor/M749153916/board.do?act=detail&deleteAt=N&idx=cb00d1a0008a85e71a41b8741facbffe449f3b764132bc8324aebf94beb102a6",
    "wonmi_transport": "https://www.best.or.kr/fmcs/289",
    "wonmi_schedule": "https://wonmi.bucheon.go.kr/site/program/board/basicboard/view?boardid=1209247&boardtypeid=28300&menuid=170005017006001&pagesize=10",
    "wonmi_2026": "https://news.bucheon.go.kr/home/kor/M749153916/board.do?act=detail&deleteAt=N&idx=cb00d1a0008a85e71a41b8741facbffe431bafabae2b14835799748c3826c8dd",
    "exclude_wonmi_2025_09_13": "https://www.theminjoo.kr/main/sub/news/view.php?brd=230&post=1213490&search=&sno=500",
    "namdong_2026": "https://www.kihoilbo.co.kr/news/articleView.html?idxno=3035988",
    "namdong_time_2026": "https://www.dreamtown.or.kr/m/main.php?board_mode=list&board_mode=view&board_no=&board_no=4586&board_page=1&board_search_head_word=&board_search_keyword=&m1=5&m2=28&m3=",
    "health_run_2025": "https://www.photosports.co.kr/Marathon/Schedule/View.aspx?idx=928",
    "health_run_2026": "https://www.youtube.com/watch?v=O6n4SMWYfxM",
}
EXTRA_EXCLUDED = {3754: ("2025-09-13",), 3226: ("2025-09-20", "2025-09-27", "2025-09-28",
                      "2025-10-10", "2025-10-11", "2025-10-12", "2025-10-19", "2025-10-25", "2026-06-14")}


def load(args, guard_report):
    raw = pd.read_csv(args.team / "data/train.csv", usecols=[*KEY, "boardings"], parse_dates=["date"])
    protected = protected_days(args.work_dir)
    exact = load_replay(args, protected)
    parts = []
    checks = {}
    for label, frame in exact.groupby("label", sort=False):
        origin = frame.origin.iloc[0]
        weight, _ = fit_weight(exact, origin)
        if not np.isclose(weight, guard_report[label]["chronological_weight"], rtol=0., atol=1e-12):
            raise ValueError(f"Chronological weight does not match the guarded report for {label}.")
        amplitude = guarded_weight(frame.date, origin, weight)
        prediction = (frame.baseline + (amplitude / BASE_WEIGHT - 1.) * frame.log_delta).clip(lower=0.)
        tested = ~frame.station_id.isin((3107, 3108, 3109)) if label == "summer" else np.ones(len(frame), dtype=bool)
        expected_key = "strict_established_68" if label == "summer" else label
        guarded = guard_report[expected_key]["guarded"]
        actual = frame.y_log.to_numpy()[tested]
        observed_rmsle = float(np.sqrt(np.mean(np.square(actual - prediction.to_numpy()[tested]))))
        if not np.isclose(observed_rmsle, guarded["candidate"], rtol=0., atol=1e-10):
            raise ValueError(f"Exact guarded validation did not reproduce for {label}.")
        scope = frame.station_id.isin(LOCAL)
        parts.append(frame.loc[scope, [*KEY, "y_log", "label", "origin", "baseline"]]
                     .assign(guarded=prediction.loc[scope]))
        checks[label] = {"verified_rmsle": observed_rmsle, "weight": weight,
                         "scope": "existing 68 stations" if label == "summer" else "all stations"}
    archive = pd.concat(parts, ignore_index=True).merge(raw.loc[raw.station_id.isin(LOCAL)], on=KEY,
                                                       how="left", validate="many_to_one")
    if not np.allclose(archive.y_log, np.log1p(archive.boardings), rtol=0., atol=1e-8):
        raise ValueError("All analog labels must match official historical observations.")
    with ZipFile(args.work_dir / "sincon_local_repro.zip") as bundle:
        prefix = "external_inputs/external/"
        weather = pd.read_csv(bundle.open(prefix + "weather_hourly.csv"), parse_dates=["date"],
                              usecols=["weather_zone", "date", "hour", "temp", "precip"])
        zones = pd.read_csv(bundle.open(prefix + "weather_zone_map.csv")).set_index("station_id").weather_zone
    return archive, protected, weather, zones, checks


def for_prediction(archive, column):
    return archive.assign(residual=archive.y_log - archive[column], predicted=np.expm1(archive[column]))

def compare(label, station, when, starts, controls_start, archive, protected, weather, zones):
    day = pd.Timestamp(when)
    count_range = range(*starts)
    shoulders = [h for h in range(max(starts[0] - 2, 6), starts[1] + 2) if h not in count_range]
    slice_ = archive.loc[archive.label.eq(label)]
    history = slice_.loc[slice_.station_id.eq(station)].copy()
    peers = slice_.loc[slice_.station_id.isin(NEARBY[station])]
    peers = peers.loc[~pd.MultiIndex.from_frame(peers[KEY[:2]]).isin(protected[1])]
    neighborhood = peers.groupby(["date", "hour"]).residual.agg(["median", "size"])
    history = history.join(neighborhood, on=["date", "hour"])
    history["relative_residual"] = history.residual - history["median"].where(history["size"].ge(2))
    origin = history.origin.iloc[0]
    if not origin < day <= pd.Timestamp("2026-06-30"):
        raise ValueError("Historical analog only.")
    holidays, events = protected
    mask = history.date.dt.day_of_week.eq(day.day_of_week) & history.date.ge(controls_start) & history.date.lt(day)
    mask &= ~history.date.isin(holidays) & ~history.date.isin(pd.DatetimeIndex(EXTRA_EXCLUDED[station]))
    mask &= ~pd.MultiIndex.from_frame(history[KEY[:2]]).isin(events)
    sampled = history.loc[mask | history.date.eq(day)]
    outputs = {}
    for date, block in sampled.groupby("date"):
        slot = block.loc[block.hour.isin(count_range)]
        surrounding = block.loc[block.hour.isin(shoulders)]
        point_weather = weather.loc[weather.weather_zone.eq(zones[station]) & weather.date.eq(date)
                                    & weather.hour.isin(count_range)]
        outputs[str(date.date())] = {
            "observed_boardings": int(slot.boardings.sum()), "predicted_boardings": float(slot.predicted.sum()),
            "log_residual": float(slot.residual.mean()), "shoulder_log_residual": float(surrounding.residual.mean()),
            "relative_residual": float(slot.relative_residual.mean()),
            "relative_shoulder": float(surrounding.relative_residual.mean()),
            "relative_hours": {str(row.hour): round(row.relative_residual, 5) for row in block.itertuples()},
            "event_window_precip_mm": float(point_weather.precip.sum()),
            "event_window_mean_temp_c": float(point_weather.temp.mean()),
            "hourly": {str(row.hour): (int(row.boardings), round(row.predicted, 4), round(row.residual, 5))
                       for row in block.itertuples()},
        }
    event = outputs[when]
    control_rows = [value for date, value in outputs.items() if date != when]
    dry = [value for value in control_rows if value["event_window_precip_mm"] <= .1]
    matched = dry if event["event_window_precip_mm"] <= .1 else [value for value in control_rows if value not in dry]
    def excess(comparisons):
        if not comparisons:
            return None
        median = float(np.median([x["log_residual"] for x in comparisons]))
        hourly = {str(hour): event["hourly"][str(hour)][2] -
                  float(np.median([row["hourly"][str(hour)][2] for row in comparisons]))
                  for hour in count_range}
        adjusted = event["log_residual"] - event["shoulder_log_residual"] - float(np.median(
            [x["log_residual"] - x["shoulder_log_residual"] for x in comparisons]))
        relative_hourly = {str(hour): event["relative_hours"][str(hour)] -
                          float(np.nanmedian([row["relative_hours"][str(hour)] for row in comparisons]))
                          for hour in count_range}
        relative = event["relative_residual"] - float(np.nanmedian([x["relative_residual"] for x in comparisons]))
        return {"days": len(comparisons), "neighbor_difference_in_differences": relative,
                "neighbor_hourly_difference_in_differences": relative_hourly, "median_boardings": float(np.median([x["observed_boardings"] for x in comparisons])),
                "median_residual": median, "event_minus_controls": event["log_residual"] - median,
                "shoulder_adjusted_event_minus_controls": adjusted, "hourly_contrast": hourly}
    return {"origin": str(origin.date()), "dates": outputs, "event_date": when, "station": station,
            "all_ordinary": excess(control_rows), "same_wet_status": excess(matched)}


def stage_wonmi(args, tests, protected, guard_report):
    analogs = [tests["wonmi_same_forecast"], tests["wonmi_closer_forecast"]]
    dates = pd.DatetimeIndex(["2026-10-17"])
    hours = [13, 14, 15, 16]
    attendance_ratio = 1700 / 2000
    shrink = .5
    cap = .10
    def bounded(hour):
        key = str(hour)
        evidence = [sample["dates"][sample["event_date"]]["hourly"][key][2] for sample in analogs]
        evidence += [sample["all_ordinary"]["neighbor_hourly_difference_in_differences"][key] for sample in analogs]
        return float(np.clip(shrink * attendance_ratio * min(evidence), 0., cap))
    grid = pd.read_csv(args.team / "data/test.csv", parse_dates=["date"], usecols=["example_id", *KEY])
    if len(grid) != 157194 or not grid.example_id.is_unique or grid.duplicated(KEY).any():
        raise ValueError("The final delta must match the full official test grid.")
    baseline_path = args.baseline.resolve()
    digest = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    if digest(baseline_path) != guard_report["export"]["sha256"]:
        raise ValueError("Expected the exact guarded submission recorded in the report.")
    baseline = pd.read_csv(baseline_path, float_precision="round_trip")
    if not baseline.example_id.equals(grid.example_id):
        raise ValueError("The incumbent does not match the official grid.")
    final = grid.loc[grid.station_id.eq(3754) & grid.date.isin(dates) & grid.hour.isin(hours)]
    if len(final) != 4 or final.date.isin(protected[0]).any() or pd.MultiIndex.from_frame(final[KEY[:2]]).isin(protected[1]).any():
        raise ValueError("The localized event must not override another protected layer.")
    delta = final.assign(log_delta=final.hour.map({hour: bounded(hour) for hour in hours}))
    if not delta.log_delta.between(1e-12, cap).all():
        raise ValueError("The late-afternoon adjustment lacks positive analog support.")
    predictions = baseline.boardings.loc[delta.index]
    vector_check = None
    comparison_check = None
    if args.vector is not None:
        vector = pd.read_csv(args.vector, float_precision="round_trip")
        event_day = grid.station_id.eq(3754) & grid.date.isin(dates)
        if (not vector.example_id.equals(grid.example_id) or int(event_day.sum()) != 18
                or not np.array_equal(vector.boardings.loc[event_day], baseline.boardings.loc[event_day])):
            raise ValueError("The vector incumbent must preserve all 18 hours of the event day.")
        vector_check = {"file": str(args.vector.resolve()), "sha256": digest(args.vector),
                        "unchanged_event_day_hours": int(event_day.sum())}
    if args.comparison is not None:
        comparison = pd.read_csv(args.comparison, float_precision="round_trip")
        if (not comparison.example_id.equals(grid.example_id) or not np.isfinite(comparison.boardings).all()
                or comparison.boardings.lt(0).any()):
            raise ValueError("The comparison incumbent must cover the official test keys and have valid predictions.")
        from_count = comparison.boardings.loc[delta.index]
        to_count = np.expm1(np.log1p(from_count) + delta.log_delta)
        comparison_check = {"file": str(args.comparison.resolve()), "sha256": digest(args.comparison),
                            "baseline_boardings": from_count.to_list(), "adjusted_boardings": to_count.to_list(),
                            "additional_boardings": float((to_count - from_count).sum())}
    path = args.delta.resolve()
    delta[["example_id", *KEY, "log_delta"]].to_csv(path, index=False, float_format="%.10f")
    revised = np.expm1(np.log1p(predictions) + delta.log_delta)
    diagnostic = {}
    for name in ("wonmi_same_forecast", "wonmi_closer_forecast"):
        sample = tests[name]
        case = sample["dates"][sample["event_date"]]
        errors = np.array([case["hourly"][str(hour)][2] for hour in hours])
        diagnostic[name] = [float(np.mean(errors ** 2) ** .5), float(np.mean((errors - delta.log_delta.to_numpy()) ** 2) ** .5)]
    return {
        "file": str(path), "sha256": digest(path), "source_analog_count": 1, "source_analog_date": "2025-09-27",
        "final_date": "2026-10-17", "station": 3754, "hours": hours,
        "attendance_ratio": attendance_ratio, "shrink": shrink, "cap": cap,
        "final_incumbent": str(baseline_path), "final_incumbent_sha256": digest(baseline_path),
        "vector_check": vector_check, "comparison_check": comparison_check, "changed_rows": len(delta), "incumbent_boardings": predictions.to_list(), "candidate_boardings": revised.to_list(),
        "analog_residual_rms_before_after_same_case_descriptive": diagnostic,
        "namdong_adoption": "excluded; 09h weather-matched spatial contrast unstable and 10–12h residual signs change",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "archive", "tabicl_oof", "retail_oof", "summer_tabicl"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--guard-report", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--vector", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--delta", type=Path)
    args = parser.parse_args()
    args.guard_report = args.guard_report or args.work_dir / "sincon_tabicl_day_weight_results.json"
    args.baseline = args.baseline or args.work_dir / "submission_station_profiles_chuseok_geomdan_tabicl_calendar_weight.csv"
    args.results = args.results or args.work_dir / "sincon_wonmi_namdong_audit_results.json"
    args.delta = args.delta or args.work_dir / "sincon_wonmi_sports_day_final_delta.csv"
    guard_report = json.loads(args.guard_report.read_text())
    archive, protected, weather, zones, checks = load(args, guard_report)
    tests = {
        "wonmi_same_forecast": ("summer", 3754, "2025-09-27", (10, 17), "2025-07-20"),
        "wonmi_closer_forecast": ("2025-08-31", 3754, "2025-09-27", (10, 17), "2025-09-01"),
        "namdong_2025_same_forecast": ("summer", 3226, "2025-09-20", (9, 13), "2025-07-20"),
        "namdong_2025_closer_forecast": ("2025-08-31", 3226, "2025-09-20", (9, 13), "2025-09-01"),
        "namdong_2026_same_forecast": ("A", 3226, "2026-06-14", (9, 13), "2026-04-26"),
    }
    def outcomes(column):
        predictions = for_prediction(archive, column)
        return {test: compare(*params, predictions, protected, weather, zones) for test, params in tests.items()}

    out = outcomes("guarded")
    baseline = outcomes("baseline")
    delta = stage_wonmi(args, out, protected, guard_report)
    comparisons = {}
    for name, result in out.items():
        event, reference = result["dates"][result["event_date"]], baseline[name]["dates"][result["event_date"]]
        comparisons[name] = {predictor: {"boardings": entry["predicted_boardings"], "log_residual": entry["log_residual"],
                                         "shoulder_log_residual": entry["shoulder_log_residual"],
                                         "spatial_contrast": comparison["all_ordinary"]["neighbor_difference_in_differences"]}
                             for predictor, entry, comparison in (("guarded", event, result),
                                                                  ("ten_percent", reference, baseline[name]))}
    args.results.write_text(json.dumps({"guarded_replay_checks": checks, "tests": out,
                                       "guarded_vs_ten_percent": comparisons, "sources": SOURCES,
                                       "wonmi_delta": delta}, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"guarded_vs_ten_percent": comparisons, "wonmi_delta": delta}, indent=2))


if __name__ == "__main__":
    main()
