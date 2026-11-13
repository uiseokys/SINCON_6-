"""Rebuild complementary blend components from their original feature inputs."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sincon_complementary_blend import JOIN, REQUIRED_GAIN, TARGET, historical_station_shift, peak_change, rmsle
from sincon_conditional_candidate import by_station_day
from sincon_festival_experiment import local_days
from sincon_holiday_transfer import transfer
from sincon_housing_growth import forecast_changes, read_exposure
from sincon_housing_occupancy import exposure_panel
from sincon_joint_candidate import setup
from sincon_local_blend_experiment import SOURCES, event_keys
from sincon_matched_year import predict as annual_forecast
from sincon_regime_anchors import correction as structural_break
from sincon_stadium_concert_analogs import read_inputs as concert_inputs
from sincon_station_state import KEY, WINDOWS, read_team
from sincon_weekly_variability import correction as weekly_correction
from sincon_weekly_variability import past_episodes, protected_days, weekly_signal


FORECAST_ORIGIN = pd.Timestamp("2026-06-30")
REPLAY_TOLERANCE = 1e-8


class Replay:
    """Keep the frozen data and feature preparation shared across origins."""

    def __init__(self, work, team, external, earlier, threads):
        self.work, self.team = work, team
        scoreboard = work / "sincon_kbo_game_data.csv"
        festivals = work / "sincon_festival_station_days.csv"
        football = work / "sincon_bucheon_football_data.csv"
        housing = work / "sincon_housing_approvals.csv"
        self.panel, _, _, self.features, self.stations, _, self.fit_tree = setup(
            team, external, scoreboard, work / "mobility_monthly", work / "sincon_station_addresses.csv", threads
        )
        self.train = pd.read_csv(team / "data/train.csv", parse_dates=["date"])
        _, self.meta, self.calendar, _ = read_team(team, external)
        centers = pd.read_csv(work / "sincon_station_centers.csv").set_index("station_id").reindex(self.meta.index)
        complexes = pd.read_csv(housing, parse_dates=["approval_date"], dtype={"first_month": str}).set_index("complex_id")
        complexes["first_month"] = pd.to_datetime(complexes.first_month, format="%Y%m", errors="coerce") + pd.Timedelta(days=14)
        walks = pd.read_csv(work / "sincon_housing_walking_distance.csv").pivot(
            index="station_id", columns="complex_id", values="walk_m"
        )
        self.old_exposure = read_exposure(housing, centers)
        # walking_ramp_full uses zero utility weight. Its deterministic
        # 60-day ramp is identical at all origins.
        self.walk_exposure, _ = exposure_panel(
            complexes, centers, pd.DataFrame(columns=["complex_id", "month", "ratio"]),
            self.old_exposure.index, FORECAST_ORIGIN, FORECAST_ORIGIN, 0., walks,
        )
        self.bus = pd.read_csv(work / "sincon_seoul_bus_daily.csv.gz", parse_dates=["date"]).set_index(KEY[:2])
        self.festival_days = pd.MultiIndex.from_frame(local_days(festivals)[KEY[:2]].drop_duplicates())
        self.local_events = event_keys(festivals, scoreboard, football)
        self.holidays, self.protected_events = protected_days(work)
        self.current, self.earlier = past_episodes(work / "sincon_local_validation.csv.gz", earlier,
                                                   self.holidays, self.protected_events)
        self.concert = concert_inputs(work, team)

    def predict(self, frame, origin, end, allow_short_history=False):
        """Rebuild log forecasts using only labels available at this origin."""
        origin, end = pd.Timestamp(origin), pd.Timestamp(end)
        frame = frame[[*KEY, "base"]].sort_values(KEY).reset_index(drop=True)
        if frame.date.min() != origin + pd.Timedelta(days=1) or frame.date.max() != end:
            raise ValueError("The baseline must cover the full declared forecast window.")
        keys = frame[KEY]
        base = frame.base.to_numpy()
        train = self.train.loc[self.train.date.le(origin)]
        panel = self.panel.loc[self.panel.date.le(end)].join(self.bus, on=KEY[:2], validate="many_to_one")
        history = panel.loc[panel.date.le(origin)]
        future = panel.loc[panel.date.gt(origin)].assign(boardings=np.nan)
        if not future[KEY].reset_index(drop=True).equals(keys):
            raise ValueError("Forecast feature keys differ from the incumbent grid.")

        prior_housing, _, _ = forecast_changes(train, keys, origin, self.old_exposure, self.calendar, self.meta)
        walk_housing, _, _ = forecast_changes(train, keys, origin, self.walk_exposure, self.calendar, self.meta)
        housing = walk_housing - .5 * prior_housing
        holiday, _ = transfer(train, frame, self.calendar, self.meta, "base", origin)
        holiday = holiday["pooled_same_holiday"] - base
        tree = np.log1p(self.fit_tree(history, future, self.features, [0]))
        augmented = np.log1p(self.fit_tree(history, future, self.features + list(self.bus.columns), [0]))
        available = future[self.bus.columns].notna().any(axis=1).to_numpy()
        bus = .5 * by_station_day(future, np.where(available, augmented - tree, 0.))
        break_change, _, _ = structural_break(train, keys, base, origin, self.meta, self.calendar, prior_housing)
        place = pd.MultiIndex.from_frame(keys[KEY[:2]])
        ordinary_place = ~place.isin(self.local_events)
        ordinary_date = keys.date.isin(self.calendar.index[self.calendar.is_holiday.eq(0)]).to_numpy()
        change = housing + holiday + ordinary_date * bus
        annual = np.zeros(len(frame))
        annual_available = (origin - self.calendar.index.min()).days >= 373
        if annual_available:
            matched = annual_forecast(history, future, self.stations, origin).to_numpy()
            annual = .35 * np.where(np.isfinite(matched) & ~place.isin(self.festival_days), matched - base, 0.)
            change += annual
        structural = np.where(ordinary_place & ordinary_date & (np.abs(housing) < .01), break_change, 0.)

        prior_blend = base + ordinary_place * change + structural
        known = self.earlier.loc[self.earlier.origin.lt(origin) & self.earlier.date.le(origin)]
        if allow_short_history and known.loc[known.eligible, "date"].nunique() < 75:
            weekly = np.zeros(len(frame))
        else:
            weekly_frame = weekly_signal(frame, self.holidays, self.protected_events)
            weekly = weekly_correction(self.earlier, weekly_frame, origin)[0].to_numpy()
        peak = peak_change(frame, self.concert, origin)
        concert_days = pd.MultiIndex.from_frame(self.concert[2][KEY[:2]])
        broad_change = np.where(place.isin(concert_days), 0., prior_blend - base)
        prediction = base + broad_change + weekly + peak
        if not np.isfinite(prediction).all():
            raise ValueError("The blend produced nonfinite forecasts.")
        components = {"housing": housing, "holiday": holiday, "bus": bus, "regime": break_change}
        if annual_available:
            components["annual"] = annual
        active = ordinary_place & ~place.isin(concert_days)
        applied = {
            "housing": np.where(active, housing, 0.),
            "holiday": np.where(active, holiday, 0.),
            "bus": np.where(active & ordinary_date, bus, 0.),
            "annual": np.where(active, annual, 0.),
            "regime": np.where(active, structural, 0.),
            "weekly_and_peak": weekly + peak,
        }
        return frame.assign(fixed_blend=prediction, **applied), components


def verify_fold(replay, fold, expected):
    origin, end = WINDOWS[fold]
    current = replay.current[fold].sort_values(KEY).reset_index(drop=True)
    rebuilt, components = replay.predict(current, origin, end)
    keys = pd.MultiIndex.from_frame(rebuilt[KEY])
    previous = expected.loc[fold].reindex(keys)
    measured = np.log1p(replay.train.set_index(KEY).boardings.reindex(keys).to_numpy())
    if (not np.array_equal(previous.y_log.to_numpy(), current.y_log.to_numpy())
            or not np.allclose(measured, current.y_log.to_numpy(), atol=1e-8, rtol=0)):
        raise ValueError(f"The replay target logs differ in fold {fold}.")
    difference = rebuilt.fixed_blend.to_numpy() - previous.fixed_blend.to_numpy()
    truth = current.y_log.to_numpy()
    errors = {}
    for name, vector in components.items():
        file, column = SOURCES[name]
        old = pd.read_csv(replay.work / file, parse_dates=["date"], usecols=[*JOIN, "reference", column])
        old = old.loc[old.fold.eq(fold)].set_index(KEY).reindex(keys)
        errors[name] = float(np.abs(vector - (old[column] - old.reference).to_numpy()).max())
    return {
        "rows": len(rebuilt),
        "baseline_rmsle": rmsle(truth, rebuilt.base.to_numpy()),
        "archived_rmsle": rmsle(truth, previous.fixed_blend.to_numpy()),
        "rebuilt_rmsle": rmsle(truth, rebuilt.fixed_blend.to_numpy()),
        "max_log_difference": float(np.abs(difference).max()),
        "component_max_log_differences": errors,
        "row_replay_verified": bool(np.isfinite(difference).all() and np.abs(difference).max() <= REPLAY_TOLERANCE),
    }


def eligible(reports):
    return (
        set(reports) == set(WINDOWS)
        and all(result["row_replay_verified"] for result in reports.values())
        and min(result["rebuilt_rmsle"] for result in reports.values()) <= TARGET
        and sum(result["baseline_rmsle"] - result["rebuilt_rmsle"] >= REQUIRED_GAIN for result in reports.values()) >= 2
    )


def evaluate_additional_origin(replay, path):
    frame = pd.read_csv(path, parse_dates=["date", "origin"],
                        usecols=[*KEY, "origin", "y_log", "local_stack"]).sort_values(KEY).reset_index(drop=True)
    if frame.origin.nunique() != 1 or frame.duplicated(KEY).any():
        raise ValueError(f"The additional forecast must contain one origin and unique keys: {path}")
    keys = pd.MultiIndex.from_frame(frame[KEY])
    actual = replay.train.set_index(KEY).boardings.reindex(keys).to_numpy()
    if not np.allclose(np.log1p(actual), frame.y_log, atol=1e-8, rtol=0.):
        raise ValueError(f"Additional forecast labels do not match the raw training counts: {path}")
    origin = frame.origin.iloc[0]
    end = frame.date.max()
    rebuilt, _ = replay.predict(frame.rename(columns={"local_stack": "base"}), origin, end)
    shift, prior_days = historical_station_shift(replay.earlier, frame, origin)
    base, estimate = (rebuilt[name].to_numpy() for name in ("base", "fixed_blend"))
    truth = frame.y_log.to_numpy()
    return {
        "origin": str(origin.date()), "end": str(end.date()), "rows": len(frame),
        "baseline_rmsle": rmsle(truth, base), "rebuilt_rmsle": rmsle(truth, estimate),
        "prior_calibrated_baseline_rmsle": rmsle(truth, base + shift),
        "prior_calibrated_blend_rmsle": rmsle(truth, estimate + shift),
        "prior_calibration_days": prior_days, "supplemental_overlapping_origin": True,
    }


def write_final(replay, destination):
    data = pd.read_csv(replay.team / "data/test.csv", parse_dates=["date"])
    base = pd.read_csv(replay.work / "submission_local_stack.csv", float_precision="round_trip")
    calibrated = pd.read_csv(replay.team / "submission_local_stack_level_station.csv", float_precision="round_trip")
    if (len(data) != 157_194 or not data.example_id.is_unique
            or not data.example_id.equals(base.example_id) or not data.example_id.equals(calibrated.example_id)):
        raise ValueError("The supplied incumbent forecast must match all final test IDs.")
    origin = FORECAST_ORIGIN
    forecast = data[KEY].assign(base=np.log1p(base.boardings))
    rebuilt, _ = replay.predict(forecast, origin, data.date.max())
    aligned = rebuilt.set_index(KEY).reindex(pd.MultiIndex.from_frame(data[KEY]))
    adjustment = aligned.fixed_blend.to_numpy() - aligned.base.to_numpy()
    result = np.expm1(np.maximum(np.log1p(calibrated.boardings.to_numpy()) + adjustment, 0.))
    if not np.isfinite(result).all() or (result < 0.).any():
        raise ValueError("The final forecasts must be finite nonnegative counts.")
    if destination.exists():
        raise FileExistsError(destination)
    data[["example_id"]].assign(boardings=result).to_csv(destination, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("earlier", type=Path)
    parser.add_argument("team_dir", type=Path)
    parser.add_argument("external_zip", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--folds", nargs="*", choices=sorted(WINDOWS), default=list(WINDOWS))
    parser.add_argument("--extra-origin", type=Path, action="append", default=[])
    parser.add_argument("--results", type=Path)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    replay = Replay(args.work_dir, args.team_dir, args.external_zip, args.earlier, args.threads)
    reports = {}
    if args.folds:
        expected = pd.read_csv(args.work_dir / "sincon_complementary_blend_validation.csv.gz",
                               parse_dates=["date"], usecols=[*JOIN, "y_log", "fixed_blend"]).set_index(JOIN)
    for fold in args.folds:
        start = perf_counter()
        reports[fold] = verify_fold(replay, fold, expected)
        reports[fold]["seconds"] = round(perf_counter() - start, 2)
        print(json.dumps({fold: reports[fold]}, allow_nan=False), flush=True)
    additional = {}
    for origin_file in args.extra_origin:
        start = perf_counter()
        entry = evaluate_additional_origin(replay, origin_file)
        entry["seconds"] = round(perf_counter() - start, 2)
        additional[entry["origin"]] = entry
        print(json.dumps({entry["origin"]: entry}, allow_nan=False), flush=True)
    accepted = eligible(reports)
    path = args.results or args.work_dir / "sincon_complementary_replay_results.json"
    path.write_text(json.dumps({
        "folds": reports, "additional_origins": additional,
        "accepted": accepted, "row_replay_tolerance": REPLAY_TOLERANCE,
    }, indent=2, allow_nan=False) + "\n")
    if args.submission:
        if not accepted:
            raise ValueError("Export requires three verified folds, 0.150 or lower and two gains of at least 0.003.")
        write_final(replay, args.submission)


if __name__ == "__main__":
    main()
