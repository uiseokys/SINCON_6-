"""Recompute downstream ordinary-day corrections from a clean forecast."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_dongchun_retail import contrasts as retail_contrasts, final_deltas as retail_deltas, inputs as retail_inputs
from sincon_dongchun_retail import run_case as retail_case, select as retail_select
from sincon_geomdan_morning_anchor import forecast as morning_forecast
from sincon_lunar_network_level import calendars, proposal as lunar_proposal, verified_lunar_phases
from sincon_munhak_ordinary import WARM, home_dates
from sincon_network_level_audit import STABLE_EXCLUSIONS, ordinary
from sincon_songdo_schedule import save_submission
from sincon_station_profile_candidate import station_adjustments
from sincon_station_state import KEY
from sincon_tabicl_daily_deviation import deviations
from sincon_tabicl_day_weight import BASE_WEIGHT, exclude_retail, fit_weight, guarded_weight
from sincon_vector_shape_weights import offsets as vector_offsets, read_source as vector_archive
from sincon_weekly_variability import protected_days


ORIGIN = pd.Timestamp("2026-06-30")
VECTOR_SHA256 = "ce830ea0813e0907667a59ba3bfd9899c2a26e0979ac6985d492dab49bbf2ea0"
DAY = KEY[:2]
OOF_KEY = ["label", "origin", *KEY]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_oof(path, history, forecast_column):
    if path.suffix == ".pkl":
        rows = pd.read_pickle(path)
    else:
        rows = pd.read_csv(path, parse_dates=["date", "origin"], float_precision="round_trip")
    rows = rows.assign(baseline=rows[forecast_column])
    if rows.duplicated(OOF_KEY).any() or not rows.date.gt(rows.origin).all() or rows.date.max() > ORIGIN:
        raise ValueError("Clean OOF forecasts must have unique keys and dates after their origin through June 2026.")
    if not np.isfinite(rows.baseline).all() or rows.baseline.lt(0.).any():
        raise ValueError("Clean OOF baseline must contain nonnegative finite log counts.")
    matched = rows.merge(history, on=KEY, how="left", validate="many_to_one")
    if matched.observed.isna().any() or ("y_log" in matched and not np.allclose(matched.y_log, matched.observed, atol=1e-8, rtol=0.)):
        raise ValueError("OOF labels must match the official training period.")
    if "reliability" not in matched:
        matched["reliability"] = 1.
    return matched.assign(y_log=matched.observed).drop(columns="observed")


def read_daily(path, keys):
    tabicl = pd.read_csv(path, parse_dates=[name for name in ("date", "origin") if name in keys])
    if "hour" in tabicl:
        if tabicl.groupby(keys).tabicl.nunique().max() != 1:
            raise ValueError("Repeated TabICL station-days must have identical predictions.")
        tabicl = tabicl.drop_duplicates(keys)
    if tabicl.duplicated(keys).any() or not np.isfinite(tabicl.tabicl).all():
        raise ValueError("The raw TabICL daily forecast must have unique finite keys.")
    return tabicl.set_index(keys).tabicl


def tabicl_signal(rows, alternative, protected):
    if not np.isfinite(alternative).all():
        raise ValueError("Each scored station-day requires a raw TabICL prediction.")
    future = rows[KEY].assign(incumbent=rows.baseline.to_numpy(), alternative=alternative)
    return deviations(future, protected) - future.incumbent.to_numpy()


def amplify_tabicl(rows, signal, weights):
    multiplier = np.where(exclude_retail(rows), 1., weights / BASE_WEIGHT)
    delta = signal * multiplier
    limit = np.divide(rows.baseline, -delta, out=np.ones(len(rows)), where=delta < 0.).clip(0., 1.)
    groups = [rows.station_id, rows.date.dt.to_period("M"), rows.date.dt.day_of_week]
    return delta * pd.Series(limit, index=rows.index).groupby(groups).transform("min").to_numpy()


def rmsle(actual, forecast):
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def groups_for_scores(rows):
    groups = {label: sample for label, sample in rows.groupby("label")}
    if {"A", "C"} <= groups.keys():
        groups["disjoint_A_C"] = pd.concat([groups["A"], groups["C"]], ignore_index=True)
    if all(label in groups for label, _ in WARM):
        groups["disjoint_warm"] = pd.concat([groups[label].loc[groups[label].date.dt.month.eq(month)] for label, month in WARM])
    if "summer" in groups:
        groups["summer_established_68"] = groups["summer"].loc[~groups["summer"].station_id.isin([3107, 3108, 3109])]
    if any(group.duplicated(KEY).any() for group in groups.values()):
        raise ValueError("Each chronological comparison must have non-overlapping station-hours.")
    return groups


def oof_layer(rows, forecast):
    cases = []
    for _, part in rows.groupby("label", sort=False):
        updated = forecast(part, part.origin.iloc[0])
        cases.append(part.assign(candidate=updated))
    trials = pd.concat(cases, ignore_index=True)
    scores = {name: {"reference": rmsle(part.y_log, part.baseline), "candidate": rmsle(part.y_log, part.candidate),
                     "rows": len(part), "changed_rows": int(part.baseline.ne(part.candidate).sum())}
              for name, part in groups_for_scores(trials).items()}
    return trials.assign(baseline=trials.candidate).drop(columns="candidate"), scores


def tabicl_oof(rows, alternatives, protected):
    episodes = []
    for label, sample in rows.groupby("label", sort=False):
        keys = pd.MultiIndex.from_frame(sample[["label", "origin", *DAY]])
        alternative = alternatives.reindex(keys).to_numpy()
        covered = np.isfinite(alternative)
        if not covered.any():
            continue
        coverage = pd.Series(covered, index=sample.index).groupby(sample.date.dt.to_period("M")).nunique()
        if coverage.gt(1).any():
            raise ValueError(f"TabICL OOF coverage must include each complete scored month in {label}.")
        sample, alternative = sample.loc[covered], alternative[covered]
        if sample.origin.nunique() != 1 or not sample.groupby(DAY).size().eq(18).all():
            raise ValueError(f"The clean episode {label} must contain complete station-days and one origin.")
        signal = tabicl_signal(sample, alternative, protected)
        episodes.append(sample.assign(fixed=sample.baseline + signal, log_delta=signal))
    if not episodes:
        raise ValueError("No matching clean OOF rows for the raw TabICL forecasts.")
    trials = pd.concat(episodes, ignore_index=True)
    fit = trials.assign(baseline=trials.fixed, log_delta=np.where(exclude_retail(trials), 0., trials.log_delta))
    guarded = []
    weight_fits = {}
    for label, sample in trials.groupby("label", sort=False):
        origin = sample.origin.iloc[0]
        weight, evidence = fit_weight(fit, origin)
        later = fit.date.gt(origin)
        altered = fit.assign(y_log=fit.y_log + np.where(later, 100., 0.))
        if fit_weight(altered, origin) != (weight, evidence):
            raise ValueError("Changing later outcomes must not affect a TabICL fit.")
        schedule = guarded_weight(sample.date, origin, weight)
        delta = amplify_tabicl(sample, sample.log_delta.to_numpy(), schedule)
        guarded.append(sample.assign(guarded=sample.baseline + delta))
        weight_fits[label] = {"weight": weight, **evidence}
    trials = pd.concat(guarded, ignore_index=True)
    scores = {name: {"rows": len(sample), **{method: rmsle(sample.y_log, sample[method])
               for method in ("baseline", "fixed", "guarded")}} for name, sample in groups_for_scores(trials).items() if len(sample)}
    primary = ("disjoint_A_C", "summer_established_68")
    if "disjoint_warm" in scores:
        primary += ("disjoint_warm",)
    fixed_ok = all(name in scores and scores[name]["fixed"] < scores[name]["baseline"] for name in primary)
    guard_ok = fixed_ok and all(scores[name]["guarded"] < scores[name]["fixed"] for name in primary)
    weight, final_evidence = fit_weight(fit, ORIGIN) if guard_ok else (BASE_WEIGHT if fixed_ok else 0., {})
    return weight, {"comparisons": scores, "fixed_10pct_accepted": fixed_ok, "guarded_accepted": guard_ok,
                    "fit_by_origin": weight_fits, "final_fit": {"weight": weight, **final_evidence}}, trials


def downstream(args):
    work = Path(__file__).resolve().parent
    grid = pd.read_csv(args.team / "data/test.csv", parse_dates=["date"], usecols=["example_id", *KEY])
    base = pd.read_csv(args.clean_base, float_precision="round_trip")
    if (len(grid) != 157_194 or grid.duplicated(KEY).any() or not grid.example_id.is_unique
            or not grid.groupby(DAY).size().eq(18).all() or list(base) != ["example_id", "boardings"]
            or not base.example_id.equals(grid.example_id) or not np.isfinite(base.boardings).all()
            or base.boardings.lt(0.).any()):
        raise ValueError("Expected the clean finite forecast in complete official test order.")
    explicit = pd.read_csv(args.protected_keys, parse_dates=["date"], usecols=["example_id", *KEY]).drop_duplicates()
    if explicit.empty or len(grid.merge(explicit, on=["example_id", *KEY], validate="one_to_one")) != len(explicit):
        raise ValueError("Protected keys must match official event rows.")
    station_days = pd.MultiIndex.from_frame(explicit[DAY].drop_duplicates())
    protected_mask = pd.MultiIndex.from_frame(grid[DAY]).isin(station_days)
    holidays, known = protected_days(work)
    protected = holidays, known.union(station_days)
    history = pd.read_csv(args.team / "data/train.csv", parse_dates=["date"], usecols=[*KEY, "boardings"])
    if history.date.max() != ORIGIN or history.duplicated(KEY).any():
        raise ValueError("Expected unique official labels through June 2026 only.")
    history["observed"] = np.log1p(history.pop("boardings"))
    result = grid.assign(baseline=np.log1p(base.boardings))
    sources = {"train": args.team / "data/train.csv", "test": args.team / "data/test.csv",
               "clean_base": args.clean_base, "protected_keys": args.protected_keys, "clean_oof": args.clean_oof,
               "tabicl_daily": args.tabicl_daily, "tabicl_oof": args.tabicl_oof, "vector": args.vector,
               "vector_oof_warm": args.vector_oof_warm}
    report = {"clean_base_sha256": digest(args.clean_base), "protected_station_days": len(station_days),
              "protected_station_hours": int(protected_mask.sum()), "stages": {},
              "inputs": {name: {"file": path.name, "sha256": digest(path)} for name, path in sources.items() if path}}

    def apply(name, candidate, evidence=None):
        previous = result.baseline.to_numpy()
        candidate = np.asarray(candidate)
        changed = candidate != previous
        if not np.isfinite(candidate).all() or (candidate < -1e-12).any() or np.any(changed & protected_mask):
            raise ValueError(f"Invalid prediction or a protected event change in {name}.")
        report["stages"][name] = {"changed_rows": int(changed.sum()),
                                  "max_abs_log_delta": float(np.max(np.abs(candidate - previous)))}
        if evidence is not None:
            report["stages"][name]["evidence"] = evidence
        result["baseline"] = candidate.clip(min=0.)

    games, _ = home_dates(work)
    station, support = station_adjustments(history, result[KEY].assign(combined=result.baseline), ORIGIN, protected, games)
    apply("station_profiles", (result.baseline + station["joint"]).clip(lower=0.), support)

    oof = read_oof(args.clean_oof, history, args.oof_forecast_column) if args.clean_oof else None
    if oof is not None:
        initial_oof = oof.set_index(OOF_KEY).baseline

        def station_prediction(part, origin):
            reference = part[KEY].assign(combined=part.baseline)
            shifts, _ = station_adjustments(history.loc[history.date.le(origin)], reference, origin, protected, games)
            return (part.baseline + shifts["joint"]).clip(lower=0.)

        oof, report["station_profiles_clean_validation"] = oof_layer(oof, station_prediction)

    signal, tabicl_trials = None, None
    if args.tabicl_daily and args.tabicl_oof:
        alternatives = read_daily(args.tabicl_oof, ["label", "origin", *DAY])
        weight, validation, tabicl_trials = tabicl_oof(oof, alternatives, protected)
        report["tabicl_clean_validation"] = validation
        if weight > 0.:
            final = read_daily(args.tabicl_daily, DAY).reindex(pd.MultiIndex.from_frame(result[DAY])).to_numpy()
            signal = tabicl_signal(result, final, protected)
            weights = guarded_weight(result.date, ORIGIN, weight)
            final_signal = amplify_tabicl(result, signal, weights)
            apply("tabicl_fixed", result.baseline + signal)
            extra = final_signal - signal
            trial_values = tabicl_trials.set_index(OOF_KEY)
            fixed = trial_values.fixed.reindex(pd.MultiIndex.from_frame(oof[OOF_KEY]))
            oof["baseline"] = np.where(fixed.notna(), fixed.to_numpy(), oof.baseline.to_numpy())

    if oof is not None:
        calendar, retail_events, _ = retail_inputs(work)
        retail_events = pd.concat([retail_events, explicit[DAY]]).drop_duplicates()
        retail = oof.assign(error=oof.y_log - oof.baseline, observed=oof.y_log, base=oof.baseline,
                            eligible=ordinary(oof, protected))
        known_retail = retail_select(retail)

        def retail_prediction(part, origin):
            prior = known_retail.loc[known_retail.date.le(origin) & known_retail.origin.lt(origin)]
            prior = prior.sort_values(["reliability", "origin"]).drop_duplicates(KEY, keep="last")
            candidate, _ = retail_case(part, retail_contrasts(prior))
            return (part.baseline + candidate).clip(lower=0.)

        revised, validation = oof_layer(retail, retail_prediction)
        accepted = all(name in validation and validation[name]["candidate"] < validation[name]["reference"]
                       for name in ("disjoint_A_C", "disjoint_warm"))
        report["dongchun_retail_clean_validation"] = {"accepted": accepted, "comparisons": validation}
        if accepted:
            shifts, support = retail_deltas(retail, calendar, retail_events)
            delta = shifts.set_index(KEY).log_delta.reindex(pd.MultiIndex.from_frame(result[KEY]), fill_value=0.).to_numpy()
            apply("dongchun_retail", result.baseline + delta, support)
            oof = revised

    phases = calendars(work)
    lunar, support = lunar_proposal(history, result[[*KEY, "baseline"]], phases, ORIGIN, protected)
    apply("lunar_eve_and_day", result.baseline + verified_lunar_phases(result, lunar, phases), support)
    morning, support = morning_forecast(history, result[[*KEY, "baseline"]], ORIGIN, protected)
    apply("geomdan_morning", result.baseline + morning["relative_peak_share"], support)
    if oof is not None:
        def lunar_prediction(part, origin):
            earlier = history.loc[history.date.le(origin)]
            delta, _ = lunar_proposal(earlier, part[[*KEY, "baseline"]], phases, origin, protected)
            return (part.baseline + verified_lunar_phases(part, delta, phases)).clip(lower=0.)

        def morning_prediction(part, origin):
            earlier = history.loc[history.date.le(origin)]
            delta, _ = morning_forecast(earlier, part[[*KEY, "baseline"]], origin, protected)
            return (part.baseline + delta["relative_peak_share"]).clip(lower=0.)

        oof, report["lunar_clean_validation"] = oof_layer(oof, lunar_prediction)
        oof, report["morning_clean_validation"] = oof_layer(oof, morning_prediction)
    if signal is not None and np.any(extra):
        apply("tabicl_lead_weight", result.baseline + extra, result.assign(weight=weights).groupby(result.date.dt.month).weight.first().to_dict())
        trial_delta = tabicl_trials.set_index(OOF_KEY).eval("guarded - fixed")
        oof["baseline"] += trial_delta.reindex(pd.MultiIndex.from_frame(oof[OOF_KEY]), fill_value=0.).to_numpy()

    if args.vector:
        if digest(args.vector) != VECTOR_SHA256 and not args.allow_retrained_vector:
            raise ValueError("Use the archived independent vector model forecast.")
        vector = pd.read_csv(args.vector, parse_dates=["date"], float_precision="round_trip")
        if len(vector) != len(grid) or not vector.fold.eq("final").all():
            raise ValueError("The independent vector forecast must contain every final key.")
        eligible = ordinary(result, protected) & ~result.station_id.isin(STABLE_EXCLUSIONS)
        values = vector_offsets(result.assign(eligible=eligible), vector.set_index(KEY, verify_integrity=True).raw_delta)["20%"]
        accepted = True
        if oof is not None:
            sources = vector_archive(work)
            if args.vector_oof_warm:
                raw = pd.read_csv(args.vector_oof_warm, parse_dates=["date"], float_precision="round_trip")
                sources.update({label: part.set_index(KEY, verify_integrity=True).raw_delta for label, part in raw.groupby("label")})

            def vector_prediction(part, _):
                ordinary_rows = ordinary(part, protected) & ~part.station_id.isin(STABLE_EXCLUSIONS)
                return vector_offsets(part.assign(eligible=ordinary_rows), sources[part.label.iloc[0]])["20%"]

            usable = oof.loc[oof.label.isin(sources)]
            vector_oof, validation = oof_layer(usable, vector_prediction)
            pooled = [name for name in ("disjoint_A_C", "disjoint_warm", "summer_established_68") if name in validation]
            accepted = len(pooled) >= 1 and all(validation[name]["candidate"] < validation[name]["reference"] for name in pooled)
            report["vector_clean_validation"] = {"accepted": accepted, "comparisons": validation}
        if accepted:
            apply("vector_xgboost", values, {"weight": .20, "raw_forecast_sha256": digest(args.vector)})
            if oof is not None:
                oof = vector_oof

    if oof is not None:
        anchored = oof.assign(initial=initial_oof.reindex(pd.MultiIndex.from_frame(oof[OOF_KEY])).to_numpy())
        report["combined_clean_validation"] = {
            name: {"reference": rmsle(part.y_log, part.initial), "candidate": rmsle(part.y_log, part.baseline),
                   "rows": len(part)} for name, part in groups_for_scores(anchored).items()}
        if args.validation_archive:
            anchored[OOF_KEY + ["y_log", "initial", "baseline"]].to_csv(
                args.validation_archive, index=False, float_format="%.17g"
            )

    changed = result.baseline.ne(np.log1p(base.boardings)).to_numpy()
    values = base.boardings.to_numpy(copy=True)
    values[changed] = np.expm1(result.baseline.to_numpy()[changed])
    if not np.array_equal(values[protected_mask], base.boardings.to_numpy()[protected_mask]):
        raise ValueError("Protected event predictions must be exactly retained.")
    save_submission(args.output, base, grid, values)
    report.update({"changed_rows": int(changed.sum()), "rows": len(grid), "sha256": digest(args.output)})
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "clean_base", "protected_keys", "output", "report"):
        parser.add_argument(name, type=Path)
    for name in ("--clean-oof", "--tabicl-daily", "--tabicl-oof", "--vector", "--vector-oof-warm"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--validation-archive", type=Path)
    parser.add_argument("--oof-forecast-column", default="combined")
    parser.add_argument("--allow-retrained-vector", action="store_true")
    args = parser.parse_args()
    if bool(args.tabicl_daily) != bool(args.tabicl_oof) or (args.tabicl_daily and not args.clean_oof):
        parser.error("TabICL needs raw final and historical forecasts with clean OOF baseline forecasts.")
    downstream(args)


if __name__ == "__main__":
    main()
