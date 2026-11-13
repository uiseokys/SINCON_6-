import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "../hb_pkg"); sys.path.insert(0, "."); import hourbias_layer as hb
import rail_shift as rs
A, days = rs.A, rs.days; rm = rs.rm; CLIP = 0.15; B = rs.B; P = rs.P
D = pickle.load(open("../nnexp/data.pkl", "rb")); df = D["df"]
oof = pd.read_csv(P / "night13/measurements/downstream_oof_nested_main.csv.gz", usecols=["origin", "date", "station_id", "hour", "friday", "lunar3122"], parse_dates=["origin", "date"])
ch = oof[(oof.lunar3122 - oof.friday).abs() > 1e-12]; print("rows changed by the lunar3122 layer in past forecasts:", len(ch), "| station-days:", sorted(set(zip(ch.date.dt.strftime("%Y-%m-%d"), ch.station_id)))[:8])
lun = pd.MultiIndex.from_frame(ch[["date", "station_id"]].drop_duplicates())
te = pd.read_csv(B / "data/test.csv", parse_dates=["date"]); a = pd.read_csv(B / "FINAL_CANDIDATES/submission_night_v4_main.csv"); 
x7 = pd.read_csv(B / "CANDIDATE_SET_1113/submission_xgb7_w10_on_curvature.csv"); v0 = pd.read_csv(B / "CANDIDATE_SET_1113/submission_rail_shift_on_xgb7.csv"); hl = pd.read_csv(B / "CANDIDATE_SET_1113/submission_rail_shift_hol_on_xgb7.csv")
m = (te.station_id == 3122) & te.date.isin(pd.to_datetime(["2026-09-24", "2026-09-25"]))
print("부평삼거리 9/24~25 (36 rows) daily totals: xgb7", x7.boardings[m].sum().round(0), "| after rail shift", v0.boardings[m].sum().round(0), "| after holiday level", hl.boardings[m].sum().round(0))
for origin, hold in A.groupby("origin", sort=True):
    hold = hold.reset_index(drop=True); y = hold.y_log.to_numpy(); p = hold[hb.MODEL].to_numpy(); free = hold.free.to_numpy(); hol = hold.date.isin(days).to_numpy()
    isl = pd.MultiIndex.from_frame(hold[["date", "station_id"]]).isin(lun)
    if not isl.any(): continue
    c = rs.shift_terms(hold, origin)["all25"]; c = np.where(free & np.isfinite(c), c, 0.0); add = np.clip(np.where(hol, 0.2, 0.1) * c, -CLIP, CLIP)
    t = df.loc[df["date"].between(str(hold.date.min().date()), str(hold.date.max().date())), ["date", "station_id", "hour"]].reset_index(drop=True); t["date"] = pd.to_datetime(t["date"]); end = str(origin.date()); t["dn"] = np.load(f"../nnexp/out4/xgbnb_{end}.npy") - np.load(f"../nnexp/out4/xgb_{end}.npy")
    dn = hold.merge(t, on=hb.KEY, how="left").dn.to_numpy(); lvl = pd.Series(np.where(free, dn, 0.0)).groupby([hold.date.to_numpy(), hold.station_id.to_numpy()]).transform("mean").to_numpy(); add2 = np.where(hol, np.clip(0.5 * lvl, -CLIP, CLIP), 0.0)
    full = p + add + add2; prot = np.where(isl, p, full)
    print(origin.date(), f"lunar rows {int(isl.sum())}: rmsle on those rows main {rm(p[isl], y[isl]):.4f} | with both rail layers {rm(full[isl], y[isl]):.4f} | overall effect of protecting them {(rm(prot, y) - rm(full, y))*1e4:+.3f}e-4 | mean shift applied there {np.mean((add + add2)[isl]):+.3f}")
