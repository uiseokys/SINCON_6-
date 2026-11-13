"""Measure bounded factual corrections to the existing festival transfer."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sincon_festival_experiment import DAY, HOURS, WEATHER, historical_effects, local_days, profile
from sincon_station_state import KEY


HISTORY_END = pd.Timestamp("2026-06-30")
JUNE = pd.Timestamp("2025-06-30")
EVENTS = ("pentaport", "bucheon_comics", "bupyeong_pungmul")
BICOM_DAY = pd.Timestamp("2025-09-25")
PUNGMUL_FRIDAYS = {pd.Timestamp("2025-09-26"), pd.Timestamp("2026-10-02")}
PUNGMUL_ROAD_STATIONS = (3119, 3120)


def read_panel(team, repro):
    train = pd.read_csv(team / "data/train.csv", parse_dates=["date"], usecols=[*KEY, "boardings"])
    if train.date.max() != HISTORY_END or train.duplicated(KEY).any():
        raise ValueError("Expected official train with no observations beyond June 30, 2026.")
    with ZipFile(repro) as archive:
        prefix = "external_inputs/"
        zones = pd.read_csv(archive.open(prefix + "external/weather_zone_map.csv"))
        weather = pd.read_csv(archive.open(prefix + "external/weather_hourly.csv"), parse_dates=["date"],
                              usecols=["weather_zone", "date", "hour", "day_temp_max", "day_precip"])
        holidays = pd.read_csv(archive.open(prefix + "calendar_features.csv"), parse_dates=["date"],
                               usecols=["date", "is_holiday"])
    panel = (train.merge(zones, on="station_id", validate="many_to_one")
             .merge(weather, on=["weather_zone", "date", "hour"], validate="many_to_one", how="left")
             .merge(holidays, on="date", validate="many_to_one", how="left"))
    if len(panel) != len(train) or panel[WEATHER].isna().any().any():
        raise ValueError("Missing archived calendar or weather coverage.")
    return panel.sort_values(KEY).reset_index(drop=True)


def source_effects(panel, events, origin):
    cutoff = pd.Timestamp(origin)
    history = panel.loc[panel.date.le(cutoff)]
    if history.date.max() != cutoff:
        raise ValueError("The official train must cover the requested cutoff.")
    return historical_effects(history, events.loc[events.date.le(cutoff)])


def effect(event, donors, variant):
    if variant == "pentaport_actual_size" and event.series == "pentaport":
        event = event._replace(attend_day=160_000 / 3)
    if variant == "bicof_actual_program" and event.series == "bucheon_comics":
        event = event._replace(attend_day=100_000 / 3)
        # Keep BICOM out of background days, and omit it as a public-festival donor.
        donors = [(old._replace(attend_day=100_000 / 3) if old.series == "bucheon_comics" else old, counts)
                  for old, counts in donors if old.series != "bucheon_comics" or old.date != BICOM_DAY]
    if variant.startswith("pungmul_") and event.series == "bupyeong_pungmul":
        if event.date in PUNGMUL_FRIDAYS and event.station_id in PUNGMUL_ROAD_STATIONS:
            if variant == "pungmul_friday_late_only":
                return np.where(HOURS >= 21, 0, profile(event, donors))
            return np.zeros(len(HOURS))
        if (variant == "pungmul_venue_split" and event.date.day_of_week in (5, 6)
                and event.station_id in PUNGMUL_ROAD_STATIONS):
            donors = [(old, counts) for old, counts in donors
                      if not (old.series == "bupyeong_pungmul" and old.date in PUNGMUL_FRIDAYS)]
    return profile(event, donors)


def predict(target, events, donors, variant):
    if set(target) != {*KEY, "conditional_stack"} or any(event.date >= target.date.min() for event, _ in donors):
        raise ValueError("Historical donors must precede the unlabeled forecast rows.")
    match = events.merge(target[DAY].drop_duplicates(), on=DAY, how="inner", validate="many_to_one")
    changes = []
    for event in match.itertuples(index=False):
        key = pd.MultiIndex.from_product([[event.date], [event.station_id], HOURS], names=KEY)
        changes.append(pd.Series(effect(event, donors, variant), index=key))
    adjustment = pd.concat(changes).groupby(level=KEY).max() if changes else pd.Series(dtype=float)
    target_keys = pd.MultiIndex.from_frame(target[KEY])
    added = adjustment.reindex(target_keys, fill_value=0).to_numpy()
    return np.log1p(np.expm1(target.conditional_stack.to_numpy().clip(min=0)) + added)


def score(actual, prediction):
    return float(np.sqrt(np.square(actual - np.maximum(prediction, 0)).mean()))


def historic(panel, events, archive):
    windows = {
        "bicof_same_edition_weekend": ("2025-09-26", "2025-09-27", "2025-09-28", [3758, 3759], "bicof_actual_program"),
        "pungmul_friday_before_open": ("2025-09-25", "2025-09-26", "2025-09-26", [3118, 3119, 3120], "pungmul_friday_only"),
        "pungmul_same_edition_sunday": ("2025-09-27", "2025-09-28", "2025-09-28", [3118, 3119, 3120], "pungmul_venue_split"),
    }
    prior = source_effects(panel, events, JUNE)
    official = panel.set_index(KEY).boardings
    report = {}
    for name, (origin, first, last, stations, variant) in windows.items():
        measured = archive.loc[archive.date.between(first, last) & archive.station_id.isin(stations)].copy()
        expected = len(pd.date_range(first, last)) * len(stations) * len(HOURS)
        truth = np.log1p(official.reindex(pd.MultiIndex.from_frame(measured[KEY])).to_numpy())
        if len(measured) != expected or measured.duplicated(KEY).any() or not np.allclose(measured.y_log, truth, atol=1e-8, rtol=0):
            raise ValueError(f"The historical archive does not match the official labels for {name}.")
        target = measured[[*KEY, "conditional_stack"]]
        offset = measured.local_stack.to_numpy() - predict(target, events, prior, "original")
        donors = source_effects(panel, events, origin)
        measured["as_filed"] = predict(target, events, donors, "original") + offset
        measured["corrected"] = predict(target, events, donors, variant) + offset
        masked = panel.assign(boardings=panel.boardings.where(panel.date.le(origin), 999_999))
        after_mutation = predict(target, events, source_effects(masked, events, origin), variant) + offset
        if not np.array_equal(after_mutation, measured.corrected.to_numpy()):
            raise ValueError("Changing withheld observations affected the proposed forecasts.")
        methods = ["local_stack", "as_filed", "corrected"]
        if name == "pungmul_friday_before_open":
            measured["corrected_late_only"] = predict(target, events, donors, "pungmul_friday_late_only") + offset
            methods.append("corrected_late_only")
        measured["y_log"] = truth
        report[name] = {"origin": origin, "first": first, "last": last, "variant": variant,
                        "withheld_label_mutation_passed": True, "metrics": {}}
        for tag, selection in {"all": measured, "daytime": measured.loc[measured.hour.between(10, 20)],
                               "late": measured.loc[measured.hour.ge(21)],
                               **{str(station): measured.loc[measured.station_id.eq(station)] for station in stations}}.items():
            report[name]["metrics"][tag] = {"rows": len(selection), **{
                column: score(selection.y_log, selection[column]) for column in methods}}
    return report


def final_sensitivity(panel, events, work, team):
    grid = pd.read_csv(team / "data/test.csv", parse_dates=["date"], usecols=[*KEY, "example_id"])
    if len(grid) != 157_194 or grid.duplicated(KEY).any() or not grid.example_id.is_unique or grid.date.min() <= HISTORY_END:
        raise ValueError("Expected the unlabeled competition grid.")
    conditional = pd.read_csv(work / "submission_conditional_stack.csv").set_index("example_id").boardings
    champion = pd.read_csv(team / "submission_local_stack_level_station.csv").set_index("example_id").boardings
    if set(conditional.index) != set(grid.example_id) or set(champion.index) != set(grid.example_id):
        raise ValueError("The frozen predictions must cover exactly the official grid.")
    grid["conditional_stack"] = np.log1p(grid.example_id.map(conditional))
    grid["champion"] = np.log1p(grid.example_id.map(champion))
    all_donors = source_effects(panel, events, HISTORY_END)
    report = {}
    for series, variants in zip(EVENTS, (("pentaport_actual_size",), ("bicof_actual_program",),
                                         ("pungmul_friday_only", "pungmul_friday_late_only", "pungmul_venue_split"))):
        affected = events.loc[events.series.eq(series) & events.date.gt(HISTORY_END), DAY]
        rows = grid.merge(affected, on=DAY, validate="many_to_one")
        target = rows[[*KEY, "conditional_stack"]]
        old = predict(target, events, all_donors, "original")
        for variant in variants:
            delta = predict(target, events, all_donors, variant) - old
            changed = np.abs(delta) > 1e-12
            if not np.isfinite(delta).all():
                raise ValueError("The local transfer changes must be finite.")
            if variant in ("pungmul_friday_only", "pungmul_friday_late_only"):
                intended = rows.date.isin(PUNGMUL_FRIDAYS) & rows.station_id.isin(PUNGMUL_ROAD_STATIONS)
                if variant == "pungmul_friday_late_only":
                    intended &= rows.hour.ge(21)
                if not np.array_equal(changed, intended.to_numpy()):
                    raise ValueError("The scoped Pungmul correction must match the declared keys.")
            rows["old_boardings"] = np.expm1(rows.champion)
            rows["revised_boardings"] = np.expm1(np.maximum(rows.champion + delta, 0))
            details = rows.groupby(DAY)[["old_boardings", "revised_boardings"]].sum().reset_index()
            report[variant] = {"changed_rows": int(changed.sum()), "max_absolute_log_change": float(np.abs(delta).max()),
                               "by_station_day": json.loads(details.to_json(orient="records", date_format="iso"))}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team", type=Path)
    parser.add_argument("--work", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    events = local_days(args.work / "sincon_festival_station_days.csv")
    panel = read_panel(args.team, args.work / "sincon_local_repro.zip")
    archive = pd.read_csv(args.work / "sincon_summer_replay_validation.csv.gz", parse_dates=["date"])
    result = {"historical": historic(panel, events, archive),
              "unlabeled_2026_transfer_only_sensitivity": final_sensitivity(panel, events, args.work, args.team)}
    out = args.work / "sincon_festival_fact_check_results.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
