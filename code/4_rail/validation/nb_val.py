import sys, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, "../hb_pkg"); sys.path.insert(0, "."); import hourbias_layer as hb
import rail_shift as rs
A, days = rs.A, rs.days; rm = rs.rm; CLIP = 0.15
D = pickle.load(open("../nnexp/data.pkl", "rb")); df = D["df"]
WS = (0.25, 0.5, 0.75, 1.0); res = {}
for line in open("../nnexp/out4/folds.txt"):
    end, vs, ve = line.split(); origin = pd.Timestamp(end); hold = A[A.origin.eq(origin)].reset_index(drop=True)
    t = df.loc[df["date"].between(vs, ve), ["date", "station_id", "hour"]].reset_index(drop=True); t["date"] = pd.to_datetime(t["date"]); t["dn"] = np.load(f"../nnexp/out4/xgbnb_{end}.npy") - np.load(f"../nnexp/out4/xgb_{end}.npy")
    m = hold.merge(t, on=hb.KEY, how="left", validate="one_to_one"); assert m.dn.notna().all()
    y = m.y_log.to_numpy(); p = m[hb.MODEL].to_numpy(); free = m.free.to_numpy(); hol = m.date.isin(days).to_numpy(); r0 = rm(p, y)
    c = rs.shift_terms(m, origin)["all25"]; c = np.where(free & np.isfinite(c), c, 0.0); v0 = p + np.clip(np.where(hol, 0.2, 0.1) * c, -CLIP, CLIP); rv = rm(v0, y)
    dn = np.where(free, m.dn.to_numpy(), 0.0); r = {"V0 vs main": rv - r0}
    for w in WS:
        r[f"V0+d w{w} (vs V0)"] = rm(v0 + np.clip(w * dn, -CLIP, CLIP), y) - rv; r[f"main+d w{w} (vs main)"] = rm(p + np.clip(w * dn, -CLIP, CLIP), y) - r0
    q = v0 + np.clip(0.5 * dn, -CLIP, CLIP); late = (m.hour >= 21).to_numpy()
    r["hol rows w0.5 vs V0"] = rm(q[hol], y[hol]) - rm(v0[hol], y[hol]); r["ord rows w0.5 vs V0"] = rm(q[~hol], y[~hol]) - rm(v0[~hol], y[~hol]); r["late w0.5 vs V0"] = rm(q[late], y[late]) - rm(v0[late], y[late])
    lvl = pd.Series(dn).groupby([m.date.to_numpy(), m.station_id.to_numpy()]).transform("mean").to_numpy()
    r["level part w0.5 vs V0"] = rm(v0 + np.clip(0.5 * lvl, -CLIP, CLIP), y) - rv; r["shape part w0.5 vs V0"] = rm(v0 + np.clip(0.5 * (dn - lvl), -CLIP, CLIP), y) - rv
    r["corr(d, V0 shift)"] = float(np.corrcoef(dn[free], (v0 - p)[free])[0, 1]); res[end] = r
T = pd.DataFrame(res).T; show = T.drop(columns=["corr(d, V0 shift)"]) * 1e4; pd.set_option("display.width", 320); pd.set_option("display.max_columns", 30)
print(show.round(2).to_string()); print("corr(d, V0 shift):", T["corr(d, V0 shift)"].round(2).to_dict()); print("mean", show.mean().round(2).to_dict()); print("worse", (show > 0).sum().to_dict())
json.dump(res, open("nb_val.json", "w"), indent=1)
