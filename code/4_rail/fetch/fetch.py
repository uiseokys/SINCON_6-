"""허용 목록의 (노선명, 역명)만 남기고 원본은 지운다."""
import json, subprocess, sys, os, pandas as pd
ALLOW = {"경인선": ["백운", "동암", "간석", "부개", "송내", "중동", "부천", "소사", "역곡", "도화", "제물포", "도원", "동인천", "인천"],
         "수인선": ["인천", "신포", "숭의", "인하대", "송도", "연수", "남동인더스파크", "호구포", "인천논현", "소래포구", "월곶"]}
seq = json.load(open("seq_map.json")); keep = []
for ym, s in seq.items():
    raw = f"raw_{ym}.csv"
    if not os.path.exists(raw):
        subprocess.run(["curl", "-s", "-m", "90", "-A", "Mozilla/5.0", "-e", "https://data.seoul.go.kr/dataList/OA-12914/F/1/datasetView.do", "-d", f"infId=OA-12914&infSeq=3&seqNo=&seq={s}",
                        "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false", "-o", raw], check=True)
    size = os.path.getsize(raw)
    for enc in ("utf-8-sig", "cp949"):
        try:
            d = pd.read_csv(raw, encoding=enc, index_col=False, usecols=[0, 1, 2, 3, 4, 5], dtype={0: str}); break
        except UnicodeDecodeError:
            continue
    d.columns = ["date", "line", "station", "board", "alight", "registered"]
    ok = pd.Series(False, index=d.index)
    for ln, sts in ALLOW.items(): ok |= d.line.eq(ln) & d.station.isin(sts)
    d = d[ok].copy()                       # everything outside the allow-list is dropped here
    assert set(d.date.str[:6]) == {ym}, (ym, set(d.date.str[:6]))
    keep.append(d); os.remove(raw)
    print(ym, "bytes", size, "allowed rows", len(d), "days", d.date.nunique(), "stations", d.groupby(["line", "station"]).ngroups, flush=True)
out = pd.concat(keep, ignore_index=True); out["date"] = pd.to_datetime(out.date, format="%Y%m%d")
out.to_csv("rail_allowed.csv.gz", index=False)
g = out.groupby(["line", "station"]); cov = g.date.agg(["min", "max", "nunique"]); cov["dups"] = out.duplicated(["date", "line", "station"]).groupby([out.line, out.station]).sum()
alld = pd.date_range("2025-01-01", "2026-10-31"); cov["missing_days"] = len(alld) - cov["nunique"]
print(cov.to_string()); print("registered date range", out.registered.min(), out.registered.max())
