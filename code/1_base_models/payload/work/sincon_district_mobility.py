"""Use mobility deviations within an administrative district as local demand signals."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_mobility_experiment import BANDS, band, local_anomalies, ridge_predict
from sincon_station_state import EVENT_STATIONS, REGULAR, WINDOWS, preceding_errors, read_team


# Use common boundaries before and after Incheon's July 2026 reorganization.
DISTRICTS = {"미추홀구": "28177", "연수구": "28185", "남동구": "28200", "부평구": "28237",
             "계양구": "28245", "서구": "28260", "서해구": "28260", "검단구": "28260", "원미구": "41192"}
CODE_CHANGES = {"28275": "28260", "28290": "28260", "28110": "28125", "28140": "28125", "28155": "28125"}


def station_districts(address_csv, team_dir):
    address = pd.read_csv(address_csv, encoding="cp949", usecols=["호선", "역명", "주소"])
    meta = pd.read_csv(team_dir / "data/metaData.csv", usecols=["station_id", "station_name", "canonical_line"])
    address["canonical_line"] = address["호선"].str[0].astype(int)
    for frame, column in ((address, "역명"), (meta, "station_name")):
        frame["name"] = frame[column].str.replace(r"\(.*?\)", "", regex=True).str.replace(r"역$", "", regex=True).str.strip()
    addresses = meta.merge(address, on=["canonical_line", "name"], how="left", validate="one_to_one")
    addresses["district"] = addresses["주소"].str.extract(r"(미추홀구|연수구|남동구|부평구|계양구|서해구|검단구|원미구|서구)")[0].map(DISTRICTS)
    if len(addresses) != 71 or addresses["district"].isna().any():
        raise ValueError("Each competition station needs a mapped official address.")
    return addresses.set_index("station_id")["district"]


def mobility_design(directory, calendar, district_ids):
    files = sorted(directory.glob("*.csv.gz"))
    arrivals = pd.concat((pd.read_csv(path, dtype={"district": str}, parse_dates=["date"]) for path in files), ignore_index=True)
    arrivals["district"] = arrivals["district"].replace(CODE_CHANGES)
    arrivals["band"] = band(arrivals["hour"])
    arrivals["city"] = np.where(arrivals["district"].str.startswith("28"), "incheon", "bucheon")
    local = arrivals.groupby(["date", "band", "district", "mode"], observed=True)["arrivals"].sum().unstack("mode")
    broad = arrivals.groupby(["date", "band", "city", "mode"], observed=True)["arrivals"].sum().unstack("mode")
    good = calendar[REGULAR].eq(0).all(axis=1)
    values = []
    for district in district_ids:
        city = "incheon" if district.startswith("28") else "bucheon"
        counts = local.xs(district, level="district").reindex(broad.xs(city, level="city").index).fillna(0)
        regional = broad.xs(city, level="city").reindex(counts.index)
        for slot in BANDS:
            days = counts.xs(slot, level="band").reindex(calendar.index)
            totals = regional.xs(slot, level="band").reindex(calendar.index)
            if days.isna().any().any() or totals.isna().any().any():
                raise ValueError(f"Incomplete district coverage in {district}, {slot}.")
            district_share = np.log1p(days.to_numpy()) - np.log1p(totals.to_numpy())
            anomalies, eligible = local_anomalies(district_share, calendar.index, good.to_numpy())
            values.append(pd.DataFrame(anomalies, columns=days.columns).assign(
                date=calendar.index, district=district, band=slot, eligible=eligible))
    return pd.concat(values, ignore_index=True).set_index(["date", "district", "band"]).sort_index()


def local_response(rows, target_column, districts, calendar):
    data = rows.assign(district=rows["station_id"].map(districts), band=band(rows["hour"]))
    average = data.groupby(["date", "district", "band"], observed=True)[target_column].mean()
    centered = average - average.groupby(["date", "band"], observed=True).transform("mean")
    ordinary = calendar[REGULAR].eq(0).all(axis=1)
    parts = []
    for (district, slot), values in centered.groupby(level=["district", "band"], observed=True):
        days = pd.DatetimeIndex(values.index.get_level_values("date"))
        clean = ordinary.loc[days].to_numpy()
        difference, valid = local_anomalies(values.to_numpy()[:, None], days, clean)
        parts.append(pd.Series(difference[valid & clean, 0], index=values.index[valid & clean]))
    return pd.concat(parts).rename("response")


def apply_regression(response, design, future, districts, calendar):
    train = design.join(response).dropna(subset=["response"])
    keys = pd.MultiIndex.from_arrays([future["date"], future["station_id"].map(districts), band(future["hour"])], names=design.index.names)
    x = design.reindex(keys)
    columns = list(design.columns.drop("eligible"))
    shifts = np.zeros(len(future))
    for slot in BANDS:
        examples = train.loc[train.index.get_level_values("band") == slot]
        selected = x.index.get_level_values("band") == slot
        if len(examples) >= 30:
            values = ridge_predict(examples[columns].to_numpy(), examples["response"].to_numpy(), x.loc[selected, columns].to_numpy())
            shifts[selected] = np.clip(values, -0.12, 0.12) * 0.5
    shifts *= calendar.loc[future["date"], REGULAR].eq(0).all(axis=1).to_numpy()
    if not np.isfinite(shifts).all():
        raise ValueError("District adjustment must be finite for all forecast rows.")
    return shifts


def run(team_dir, external_zip, directory, address_csv):
    train, meta, calendar, predictions = read_team(team_dir, external_zip)
    districts = station_districts(address_csv, team_dir)
    design = mobility_design(directory, calendar, districts.unique())
    stable = meta.index[meta["first_date"].eq(meta["first_date"].min()) & ~meta.index.isin(EVENT_STATIONS)]
    report = {}
    for name, (cutoff, _) in WINDOWS.items():
        origin = pd.Timestamp(cutoff)
        history = train.loc[train["date"].le(origin) & train["station_id"].isin(stable)]
        history = history.assign(log=np.log1p(history["boardings"]))
        fold = predictions.loc[predictions["fold"].eq(name)]
        future = fold[["date", "station_id", "hour"]]
        changes = {"ensemble": np.zeros(len(fold))}
        response = local_response(history, "log", districts, calendar)
        changes["district_demand"] = apply_regression(response, design, future, districts, calendar)
        earlier = preceding_errors(predictions, origin)
        if not earlier.empty:
            response = local_response(earlier.loc[earlier["station_id"].isin(stable)], "error", districts, calendar)
            changes["district_past_residual"] = apply_regression(response, design, future, districts, calendar)
        truth = fold["y_log"].to_numpy()
        baseline = fold["ens_log"].to_numpy()
        report[name] = {key: float(np.sqrt(np.mean(np.square(truth - np.maximum(baseline + value, 0))))) for key, value in changes.items()}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team_dir", type=Path)
    parser.add_argument("external_zip", type=Path)
    parser.add_argument("mobility_directory", type=Path)
    parser.add_argument("station_addresses", type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    output = json.dumps(run(args.team_dir, args.external_zip, args.mobility_directory, args.station_addresses), indent=2, allow_nan=False) + "\n"
    if args.results:
        args.results.write_text(output, encoding="utf-8")
    print(output, end="")
