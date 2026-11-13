import sys, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, "../hb_pkg"); sys.path.insert(0, "."); import hourbias_layer as hb
import rail_shift as rs
A, RB, LV, days = rs.A, rs.RB, rs.LV, rs.days; rm = rs.rm; CLIP = 0.15; K, ALPHA = 5, 5.0
allr = pd.read_csv("rail_all_allowed.csv.gz", parse_dates=["date"]); OB = np.log1p(allr.pivot_table(index="date", columns=["line", "station"], values="board", aggfunc="sum")).sort_index()
OB = OB.loc[:, OB.notna().sum().eq(669)]; print("other-line stations used:", OB.shape[1], "| lines:", OB.columns.get_level_values(0).nunique())
def roll_anom(frame): return frame - pd.concat([frame.shift(7 * k) for k in range(1, 9)], keys=range(8)).groupby(level=1).median()
AOB = roll_anom(OB); AL = roll_anom(LV)
D = pickle.load(open("../nnexp/data.pkl", "rb")); df = D["df"]
def fit(origin):
    d = AOB.index[(AOB.index <= origin) & AOB.notna().all(axis=1).to_numpy()]; X = AOB.loc[d].to_numpy(); mu = X.mean(0); U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    V = Vt[:K].T; F = (X - mu) @ V; sd = F.std(0); F = F / sd; Y = AL.reindex(d)
    Bc = {}
    for sid in Y.columns:
        yv = Y[sid].to_numpy(); ok = np.isfinite(yv)
        if ok.sum() < 60: continue
        Bc[sid] = np.linalg.solve(F[ok].T @ F[ok] + ALPHA * np.eye(K), F[ok].T @ yv[ok])
    return mu, V, sd, Bc, float((s[:K] ** 2).sum() / (s ** 2).sum())
res = {}
for line in open("../nnexp/out4/folds.txt"):
    end, vs, ve = line.split(); origin = pd.Timestamp(end); m = A[A.origin.eq(origin)].reset_index(drop=True)
    y = m.y_log.to_numpy(); p = m[hb.MODEL].to_numpy(); hol = m.date.isin(days).to_numpy()
    lun = (m.station_id.eq(3122) & m.date.isin(pd.to_datetime(["2025-10-05", "2025-10-06", "2026-02-16", "2026-02-17"]))).to_numpy(); free = m.free.to_numpy() & ~lun
    dts = pd.DatetimeIndex(sorted(m.date.unique())); abar = (RB.reindex(dts) - rs.dow_base(RB, origin).reindex(dts.dayofweek).to_numpy()).mean(axis=1).reindex(m.date.to_numpy()).to_numpy()
    sb = rs.dow_base(LV, origin).stack().rename("b").reset_index(); sb.columns = ["dow", "station_id", "b"]
    lvl = m.groupby(["date", "station_id"])[hb.MODEL].transform("mean").to_numpy(); b = m[["date", "station_id"]].assign(dow=m.date.dt.dayofweek).merge(sb, on=["dow", "station_id"], how="left").b.to_numpy(); ahat = lvl - b
    mu, V, sd, Bc, ev = fit(origin); Z = OB.reindex(dts) - rs.dow_base(OB, origin).reindex(dts.dayofweek).to_numpy(); F = ((Z.to_numpy() - mu) @ V) / sd
    Fd = pd.DataFrame(F, index=dts); Bm = np.stack([Bc.get(s, np.full(K, np.nan)) for s in m.station_id.to_numpy()]); chat = (Fd.reindex(m.date.to_numpy()).to_numpy() * Bm).sum(1)
    c0 = np.where(free & np.isfinite(abar - ahat), abar - ahat, 0.0); v0 = p + np.clip(np.where(hol, 0.2, 0.1) * c0, -CLIP, CLIP)
    q = df.loc[df["date"].between(vs, ve), ["date", "station_id", "hour"]].reset_index(drop=True); q["date"] = pd.to_datetime(q["date"]); q["dn"] = np.load(f"../nnexp/out4/xgbnb_{end}.npy") - np.load(f"../nnexp/out4/xgb_{end}.npy")
    dn = np.where(free, m.merge(q, on=hb.KEY, how="left").dn.to_numpy(), 0.0); lv = pd.Series(dn).groupby([m.date.to_numpy(), m.station_id.to_numpy()]).transform("mean").to_numpy(); hl = np.where(hol, np.clip(0.5 * lv, -CLIP, CLIP), 0.0)
    cur = v0 + hl; rc = rm(cur, y); r0 = rm(p, y); c2 = np.where(free & np.isfinite(chat - ahat), chat - ahat, 0.0)
    true_lvl = m.groupby(["date", "station_id"]).y_log.transform("mean").to_numpy(); ok = free & np.isfinite(chat)
    r = {"current vs main": rc - r0, "var explained by 5 factors": ev, "level rmse main": rm(lvl[ok], true_lvl[ok]), "level rmse factors nowcast": rm((b + chat)[ok], true_lvl[ok]), "level rmse 25-station mean": rm((b + abar)[ok], true_lvl[ok])}
    for wo, wh in ((0.1, 0.2), (0.2, 0.3), (0.3, 0.5)):
        r[f"factors o{wo} h{wh} + holiday level"] = rm(p + np.clip(np.where(hol, wh, wo) * c2, -CLIP, CLIP) + hl, y) - rc
        r[f"factors o{wo} h{wh} (no holiday level)"] = rm(p + np.clip(np.where(hol, wh, wo) * c2, -CLIP, CLIP), y) - rc
    res[end] = r
T = pd.DataFrame(res).T; pd.set_option("display.width", 320); pd.set_option("display.max_columns", 30)
print(T[["level rmse main", "level rmse factors nowcast", "level rmse 25-station mean", "var explained by 5 factors"]].round(4).to_string())
cols = [c for c in T.columns if c.startswith("factors")]; print((T[cols] * 1e4).round(2).to_string()); print("mean", (T[cols].mean() * 1e4).round(2).to_dict()); print("worse than current", (T[cols] > 1e-9).sum().to_dict())
json.dump(res, open("factor_val.json", "w"), indent=1)
