"""Aggregate monthly Incheon bus counts around competition stations."""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


PORTAL = "https://www.data.go.kr"
SOURCE = f"{PORTAL}/data/15048264/fileData.do"
STOP_SOURCE = f"{PORTAL}/data/15067528/fileData.do"
DOWNLOAD = f"{PORTAL}/cmm/cmm/fileDownload.do"
COUNTS = ["bus_boardings", "bus_alightings"]
# These files have 13.7m and 3.5m total boardings against 19m–25m in
# neighboring months. January also has only 3,000 stops versus about 6,700.
INCOMPLETE_MONTHS = {"202510", "202601"}


def fetch(url, referer, params=None):
    headers = {"Referer": referer, "User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"}
    data = urlencode(params).encode() if params else None
    with urlopen(Request(url, data=data, headers=headers), timeout=45) as response:
        return response.read()


def download_attachment(dataset, detail, serial, referer, directory, filename):
    path = directory / filename
    if path.exists() and path.stat().st_size:
        return path
    metadata = json.loads(fetch(f"{PORTAL}/tcs/dss/selectFileDataDownload.do", referer, {
        "publicDataPk": dataset, "publicDataDetailPk": detail, "fileDetailSn": serial, "publicDataTyCode": "PR0051",
    }))
    attachment = metadata.get("atchFileId")
    if not metadata.get("status") or not attachment:
        raise ValueError(f"Missing public attachment for {filename}.")
    # The history entry's serial can differ from its attached CSV number.
    result = fetch(f"{DOWNLOAD}?" + urlencode({"atchFileId": attachment, "fileDetailSn": 1}), referer)
    if not result or result.startswith(b"<"):
        raise ValueError(f"No CSV bytes returned for {filename}.")
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(result)
    temporary.replace(path)
    return path


def monthly_files(directory, start, end):
    page = fetch(SOURCE, SOURCE).decode()
    current = re.search(r"fn_fileDataDown\('15048264', '(uddi:[^']+)',\s*'','(\d+)'", page)
    date = re.search(r"현황_(20\d{6})", page)
    if not current or not date:
        raise ValueError("Cannot read the current public bus counts entry.")
    history = fetch(f"{PORTAL}/tcs/dss/selectHistAndCsvData.do", SOURCE,
                    {"publicDataPk": "15048264", "publicDataDetailPk": current[1]}).decode()
    entries = re.findall(r'data-public-pk="(uddi:[^"]+)"\s+data-public-detail-sn="(\d+)"[^>]*>\s*인천광역시_정류장별 이용승객 현황_(20\d{6})', history)
    entries.append((current[1], current[2], date[1]))
    by_month = {ymd[:6]: (detail, serial, ymd) for detail, serial, ymd in entries}
    months = pd.period_range(start, end, freq="M").strftime("%Y%m")
    if missing := set(months) - by_month.keys():
        raise ValueError(f"Missing monthly public bus files: {sorted(missing)}")

    def download(month):
        detail, serial, _ = by_month[month]
        path = download_attachment(15048264, detail, serial, SOURCE, directory, f"{month}.csv")
        print(f"{month} {path.stat().st_size:,} bus CSV bytes", flush=True)
        return path

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(download, months))


def read_csv(path, **kwargs):
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unsupported CSV encoding in {path}.")


def clean_name(values):
    return (values.str.replace(r"\(.*?\)", "", regex=True).str.replace(r"\s+", "", regex=True)
            .str.replace(r"역$", "", regex=True).replace({"서해구청": "서구청"}))


def locations(entrances_csv, coordinates_csv, metadata):
    stops = read_csv(coordinates_csv, dtype=str).rename(columns={"모바일단축번호": "stop_id", "도시명": "city", "정류장명": "bus_stop",
                                                                 "위도": "latitude", "경도": "longitude"})
    stops = stops.loc[stops["city"].str.startswith("인천", na=False)].copy()
    stops["stop_id"] = stops["stop_id"].str.strip().str.lstrip("0")
    for column in ("latitude", "longitude"):
        stops[column] = pd.to_numeric(stops[column], errors="coerce")
    stops = stops.dropna(subset=["stop_id", "latitude", "longitude"])
    coords = ["latitude", "longitude"]
    entrances = read_csv(entrances_csv).rename(columns={"역번호": "code", "역명": "name", "위도": "latitude", "경도": "longitude"})
    entrances["station_id"] = entrances["code"] + np.where(entrances["code"].ge(700), 3002, 3000)
    entrances["name"] = clean_name(entrances["name"])
    expected = clean_name(metadata.set_index("station_id")["station_name"])
    if not entrances["station_id"].map(expected).eq(entrances["name"]).all():
        raise ValueError("The official exit numbering does not match competition station names.")
    centers = entrances.groupby("station_id")[coords].median()
    # The official exit file omits this station. Use the midpoint of the two
    # IncheonBIS bus stops that are explicitly named after its subway station.
    lake = stops.loc[stops["bus_stop"].eq("검단호수공원역"), coords]
    if 3107 not in centers.index and len(lake) == 2:
        centers.loc[3107] = lake.mean()
    centers = centers.reindex(metadata["station_id"])
    if centers.isna().any().any() or not centers["latitude"].between(37.2, 37.8).all():
        raise ValueError("Every competition station must have a valid coordinate.")
    return centers, stops.groupby("stop_id")[coords].median()


def summarize(months, entrances, coordinates, metadata):
    centers, stops = locations(entrances, coordinates, metadata)
    # Near 37.5° N, use a local meter scale for distance under one kilometer.
    scale = np.array([111_195, 111_195 * np.cos(np.deg2rad(37.5))])
    distance = cdist(centers.to_numpy() * scale, stops.to_numpy() * scale)
    weights = np.exp(-0.5 * (distance / 350) ** 2) * (distance <= 1000)
    rename = {"정류소아이디": "stop_id", "정류장번호": "stop_id", "정류소번호": "stop_id", "총승차건수": COUNTS[0], "승차인원수": COUNTS[0],
              "총하차건수": COUNTS[1], "하차인원수": COUNTS[1]}
    results = []
    for path in sorted(months.glob("202[56][01][0-9].csv")):
        month = pd.Period(path.stem, freq="M")
        if path.stem in INCOMPLETE_MONTHS:
            print(f"{month}: incomplete published file, skipping", flush=True)
            continue
        data = read_csv(path, dtype=str).rename(columns=lambda name: rename.get(name.replace(" ", ""), name.strip()))
        data["stop_id"] = data["stop_id"].str.strip().str.lstrip("0")
        for name in COUNTS:
            data[name] = pd.to_numeric(data[name].str.replace(",", ""), errors="raise")
        totals = data.groupby("stop_id")[COUNTS].sum()
        coverage = float(totals.loc[totals.index.intersection(stops.index), COUNTS[0]].sum() / totals[COUNTS[0]].sum())
        if coverage < 0.90:
            raise ValueError(f"Station coordinate matching covers only {coverage:.1%} of bus demand in {month}.")
        volumes = totals.reindex(stops.index).fillna(0).to_numpy(dtype=float) / month.days_in_month
        station_volumes = weights @ volumes
        data = pd.DataFrame(station_volumes, index=centers.index, columns=COUNTS)
        data["bus_stop_count"] = ((weights > 0) & (volumes[:, 0] > 0)).sum(axis=1)
        data["bus_city_boardings"] = volumes[:, 0].sum()
        data["bus_city_alightings"] = volumes[:, 1].sum()
        data["month"] = str(month)
        results.append(data.reset_index())
        print(f"{month}: {coverage:.1%} of bus boardings matched to stop coordinates", flush=True)
    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("entrances", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--coordinates", type=Path)
    parser.add_argument("--start", default="2025-01")
    parser.add_argument("--end", default="2026-09")
    parser.add_argument("--cached", action="store_true")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if not args.cached:
        monthly_files(args.directory, args.start, args.end)
    coords = args.coordinates or download_attachment(15067528, "uddi:ed6af349-4869-4c9a-82e1-82ecc32c686e", 1,
                                                      STOP_SOURCE, args.directory, "bus_coordinates.csv")
    summarized = summarize(args.directory, args.entrances, coords, pd.read_csv(args.metadata))
    summarized.to_csv(args.output, index=False)
