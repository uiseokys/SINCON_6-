"""7호선 전체, 7호선·인천 1·2호선과의 환승역을 빼고 나머지 노선의 일별 승차만 남긴다. 원본은 지운다."""
import json, subprocess, os, re, pandas as pd
norm = lambda s: re.sub(r"\(.*?\)", "", str(s)).strip()
seq = json.load(open("seq_map.json"))
def get(ym):
    raw = f"raw_{ym}.csv"
    subprocess.run(["curl", "-s", "-m", "90", "-A", "Mozilla/5.0", "-e", "https://data.seoul.go.kr/dataList/OA-12914/F/1/datasetView.do", "-d", f"infId=OA-12914&infSeq=3&seqNo=&seq={seq[ym]}", "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false", "-o", raw], check=True)
    for enc in ("utf-8-sig", "cp949"):
        try: d = pd.read_csv(raw, encoding=enc, index_col=False, usecols=[0, 1, 2, 3], dtype={0: str}); break
        except UnicodeDecodeError: continue
    os.remove(raw); d.columns = ["date", "line", "station", "board"]; return d
first = get("202506")                                   # a month before July 2026: used only to list the names of Line 7 stations
line7 = sorted(set(first.loc[first.line.eq("7호선"), "station"].map(norm)))
INCHEON = ["부평", "주안", "원인재", "계양", "검암", "부천종합운동장", "인천시청", "부평구청", "석남"]
BAN = set(line7) | set(INCHEON); json.dump({"line7_station_names": line7, "incheon_transfer": INCHEON}, open("ban_list.json", "w"), ensure_ascii=False, indent=1)
print("Line 7 station names:", len(line7), "| banned names:", len(BAN))
keep = []
for ym in seq:
    d = get(ym) if ym != "202506" else first
    ok = d.line.ne("7호선") & ~d.station.map(norm).isin(BAN)
    k = d[ok].copy(); keep.append(k); print(ym, "kept", len(k), "of", len(d), "| lines", k.line.nunique(), "stations", k.groupby(["line", "station"]).ngroups, flush=True)
out = pd.concat(keep, ignore_index=True); out["date"] = pd.to_datetime(out.date, format="%Y%m%d")
assert not out.line.eq("7호선").any() and not out.station.map(norm).isin(BAN).any()
out.to_csv("rail_all_allowed.csv.gz", index=False)
cov = out.groupby(["line", "station"]).date.nunique(); print("stations with all 669 days:", int((cov == 669).sum()), "of", len(cov), "| lines:", sorted(out.line.unique()))
print("transfer-station rows dropped on other lines (sample month 202506):", sorted(set(first.loc[first.line.ne("7호선") & first.station.map(norm).isin(BAN), "station"]))[:40])
