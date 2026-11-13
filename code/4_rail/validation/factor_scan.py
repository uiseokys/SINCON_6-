import sys, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, "../hb_pkg"); sys.path.insert(0, "."); import hourbias_layer as hb
import rail_shift as rs
A, RB, LV, days = rs.A, rs.RB, rs.LV, rs.days; rm = rs.rm; CLIP = 0.15; K, ALPHA = 5, 5.0
allr = pd.read_csv("rail_all_allowed.csv.gz", parse_dates=["date"]); OB = np.log1p(allr.pivot_table(index="date", columns=["line", "station"], values="board", aggfunc="sum")).sort_index()
OB = OB.loc[:, OB.notna().sum().eq(669)]; print("other-line stations used:", OB.shape[1], "| lines:", OB.columns.get_level_values(0).nunique())
def roll_anom(frame): return frame - pd.concat([frame.shift(7 * k) for k in range(1, 9)], keys=range(8)).groupby(level=1).median()
AOB = roll_anom(OB); AL = roll_anom(LV)
D = pickle.load(open("../nnexp/data.pkl", "rb")); df = D["df"]
def fit(origin, K, ALPHA):
    d = AOB.index[(AOB.index <= origin) & AOB.notna().all(axis=1).to_numpy()]; X = AOB.loc[d].to_numpy(); mu = X.mean(0); U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    V = Vt[:K].T; F = (X - mu) @ V; sd = F.std(0); F = F / sd; Y = AL.reindex(d); Bc = {}
    for sid in Y.columns:
        yv = Y[sid].to_numpy(); ok = np.isfinite(yv)
        if ok.sum() >= 60: Bc[sid] = np.linalg.solve(F[ok].T @ F[ok] + ALPHA * np.eye(K), F[ok].T @ yv[ok])
    return mu, V, sd, Bc
CONF = {"K5 a5 (current)": (5, 5.0, 0.1, 0.2), "K3": (3, 5.0, 0.1, 0.2), "K8": (8, 5.0, 0.1, 0.2), "K12": (12, 5.0, 0.1, 0.2), "K20": (20, 5.0, 0.1, 0.2), "K5 a1": (5, 1.0, 0.1, 0.2), "K5 a20": (5, 20.0, 0.1, 0.2), "K8 w.15": (8, 5.0, 0.15, 0.2), "K12 w.15": (12, 5.0, 0.15, 0.2)}
res = {}
for line in open("../nnexp/out4/folds.txt"):
    end, vs, ve = line.split(); origin = pd.Timestamp(end); m = A[A.origin.eq(origin)].reset_index(drop=True)
    y = m.y_log.to_numpy(); p = m[hb.MODEL].to_numpy(); hol = m.date.isin(days).to_numpy()
    lun = (m.station_id.eq(3122) & m.date.isin(pd.to_datetime(["2025-10-05", "2025-10-06", "2026-02-16", "2026-02-17"]))).to_numpy(); free = m.free.to_numpy() & ~lun
    dts = pd.DatetimeIndex(sorted(m.date.unique())); sb = rs.dow_base(LV, origin).stack().rename("b").reset_index(); sb.columns = ["dow", "station_id", "b"]
    lvl = m.groupby(["date", "station_id"])[hb.MODEL].transform("mean").to_numpy(); b = m[["date", "station_id"]].assign(dow=m.date.dt.dayofweek).merge(sb, on=["dow", "station_id"], how="left").b.to_numpy(); ahat = lvl - b
    q = df.loc[df["date"].between(vs, ve), ["date", "station_id", "hour"]].reset_index(drop=True); q["date"] = pd.to_datetime(q["date"]); q["dn"] = np.load(f"../nnexp/out4/xgbnb_{end}.npy") - np.load(f"../nnexp/out4/xgb_{end}.npy")
    dn = np.where(free, m.merge(q, on=hb.KEY, how="left").dn.to_numpy(), 0.0); lv = pd.Series(dn).groupby([m.date.to_numpy(), m.station_id.to_numpy()]).transform("mean").to_numpy(); hl = np.where(hol, np.clip(0.5 * lv, -CLIP, CLIP), 0.0)
    Z = (OB.reindex(dts) - rs.dow_base(OB, origin).reindex(dts.dayofweek).to_numpy()).to_numpy(); r = {}
    for name, (K_, A_, wo, wh) in CONF.items():
        mu, V, sd, Bc = fit(origin, K_, A_); Fd = pd.DataFrame(((Z - mu) @ V) / sd, index=dts); Bm = np.stack([Bc.get(s, np.full(K_, np.nan)) for s in m.station_id.to_numpy()]); chat = (Fd.reindex(m.date.to_numpy()).to_numpy() * Bm).sum(1)
        c2 = np.where(free & np.isfinite(chat - ahat), chat - ahat, 0.0); r[name] = rm(p + np.clip(np.where(hol, wh, wo) * c2, -CLIP, CLIP) + hl, y)
    res[end] = r
T = pd.DataFrame(res).T; D_ = T.sub(T["K5 a5 (current)"], axis=0).drop(columns="K5 a5 (current)") * 1e4; pd.set_option("display.width", 250)
print(D_.round(2).to_string()); print("mean", D_.mean().round(2).to_dict()); print("worse than current", (D_ > 1e-9).sum().to_dict())
json.dump(res, open("factor_scan.json", "w"), indent=1)
