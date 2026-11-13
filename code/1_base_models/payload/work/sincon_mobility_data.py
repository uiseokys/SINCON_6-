"""Aggregate Seoul–KT mobile-signal estimates for bus, walking and car arrivals."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd


SOURCE = "https://data.seoul.go.kr/dataList/OA-22655/F/1/datasetView.do"
DOWNLOAD = "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?useCache=false"
MODES = {4: "regional_bus", 5: "local_bus", 7: "walking", 8: "car"}
FIELDS = ["d_admdong_cd", "time_cd", "move_trans", "total_cnt", "etl_ymd"]


def summarize_month(path, geography="district"):
    days = []
    with ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".csv"):
                continue
            with archive.open(member) as source:
                rows = pd.read_csv(source, usecols=FIELDS, dtype={"d_admdong_cd": str})
            relevant = rows["d_admdong_cd"].str.startswith(("28", "4119"), na=False) & rows["move_trans"].isin(MODES)
            rows = rows.loc[relevant].copy()
            if rows.empty:
                raise ValueError(f"No Incheon or Bucheon observations in {member}.")
            clock = pd.to_numeric(rows["time_cd"], errors="raise").astype(int)
            subhour = clock.ge(100)
            if not (clock[subhour] % 100).isin([0, 20, 40]).all():
                raise ValueError(f"Unexpected minute code in {member}.")
            # The published peak periods use HHMM (e.g. 0720), other periods HH.
            rows["hour"] = clock.where(~subhour, clock // 100)
            rows["total_cnt"] = pd.to_numeric(rows["total_cnt"], errors="raise")
            rows[geography] = rows["d_admdong_cd"].str[:5] if geography == "district" else rows["d_admdong_cd"]
            rows["mode"] = rows["move_trans"].map(MODES)
            daily = rows.groupby(["etl_ymd", "hour", geography, "mode"], as_index=False)["total_cnt"].sum()
            days.append(daily)
    return pd.concat(days, ignore_index=True).rename(columns={"etl_ymd": "date", "total_cnt": "arrivals"})


def download_month(month, directory, geography="district"):
    output = directory / f"{month}.csv.gz"
    if output.exists():
        return output
    probe = directory.parent / "mobility_202501_probe.zip"
    with TemporaryDirectory(dir=directory) as temp:
        if month == "202501" and probe.exists():
            archive = probe
        else:
            body = urlencode({"infId": "OA-22655", "infSeq": "1", "seqNo": "", "seq": month}).encode()
            request = Request(DOWNLOAD, data=body, headers={"Referer": SOURCE, "User-Agent": "Mozilla/5.0"})
            archive = Path(temp) / "month.zip"
            with urlopen(request, timeout=90) as response, archive.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        summary = summarize_month(archive, geography)
        dates = pd.to_datetime(summary["date"].astype(str), format="%Y%m%d")
        month_days = pd.period_range(month, month, freq="M")[0].days_in_month
        if set(dates.dt.strftime("%Y%m")) != {month} or dates.nunique() != month_days:
            raise ValueError(f"Archive does not cover the complete month {month}.")
        if set(summary["hour"]) != set(range(24)) or not set(summary["mode"]).issubset(MODES.values()):
            raise ValueError(f"Unexpected time or travel mode coverage in {month}.")
        temporary = Path(temp) / "summary.csv.gz"
        summary.assign(date=dates).to_csv(temporary, index=False)
        temporary.replace(output)
    print(f"{month} {len(summary):,} {geography}-hour-mode rows", flush=True)
    return output


def run(directory, start, end, geography):
    directory.mkdir(parents=True, exist_ok=True)
    months = pd.period_range(start, end, freq="M").strftime("%Y%m")
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(lambda month: download_month(month, directory, geography), months))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--start", default="2025-01")
    parser.add_argument("--end", default="2026-10")
    parser.add_argument("--geography", choices=["district", "dong"], default="district")
    args = parser.parse_args()
    run(args.directory, args.start, args.end, args.geography)
