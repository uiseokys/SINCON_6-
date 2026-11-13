import sys, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, "../hb_pkg"); sys.path.insert(0, "."); import hourbias_layer as hb
import rail_shift as rs
A, days = rs.A, rs.days; rm = rs.rm; CLIP = 0.15
D = pickle.load(open("../nnexp/data.pkl", "rb")); df = D["df"]; res = {}
for line in open("../nnexp/out4/folds.txt"):
    end, vs, ve = line.split(); origin = pd.Timestamp(end); hold = A[A.origin.eq(origin)].reset_index(drop=True)
    t = df.loc[df["date"].between(vs, ve), ["date", "station_id", "hour"]].reset_index(drop=True); t["date"] = pd.to_datetime(t["date"]); t["dn"] = np.load(f"../nnexp/out4/xgbnb_{end}.npy") - np.load(f"../nnexp/out4/xgb_{end}.npy")
    m = hold.merge(t, on=hb.KEY, how="left", validate="one_to_one"); y = m.y_log.to_numpy(); p = m[hb.MODEL].to_numpy(); free = m.free.to_numpy(); hol = m.date.isin(days).to_numpy()
    c = rs.shift_terms(m, origin)["all25"]; c = np.where(free & np.isfinite(c), c, 0.0); v0 = p + np.clip(np.where(hol, 0.2, 0.1) * c, -CLIP, CLIP); rv = rm(v0, y)
    dn = np.where(free, m.dn.to_numpy(), 0.0); lvl = pd.Series(dn).groupby([m.date.to_numpy(), m.station_id.to_numpy()]).transform("mean").to_numpy(); r = {}
    for w in (0.25, 0.5, 0.75, 1.0):
        r[f"hol level w{w}"] = rm(np.where(hol, v0 + np.clip(w * lvl, -CLIP, CLIP), v0), y) - rv
        r[f"hol full w{w}"] = rm(np.where(hol, v0 + np.clip(w * dn, -CLIP, CLIP), v0), y) - rv
    r["ord level w0.25"] = rm(np.where(~hol, v0 + np.clip(0.25 * lvl, -CLIP, CLIP), v0), y) - rv
    # replace the V0 holiday shift by the model's holiday level instead of adding to it
    v0n = p + np.clip(np.where(hol, 0.0, 0.1) * c, -CLIP, CLIP)
    for w in (0.5, 1.0): r[f"hol level w{w} without V0 holiday shift"] = rm(np.where(hol, v0n + np.clip(w * lvl, -CLIP, CLIP), v0n), y) - rv
    r["holiday dates"] = sorted(set(m.date[hol].dt.strftime("%m-%d")))[:12]; res[end] = r
T = pd.DataFrame({k: {a: b for a, b in v.items() if a != "holiday dates"} for k, v in res.items()}).T * 1e4; pd.set_option("display.width", 320); pd.set_option("display.max_columns", 30)
print(T.round(2).to_string()); print("mean", T.mean().round(2).to_dict()); print("worse", (T > 0).sum().to_dict())
for k in ("2026-02-28", "2026-03-31", "2026-04-30"): print(k, res[k]["holiday dates"])
json.dump(res, open("nb_val2.json", "w"), indent=1)
