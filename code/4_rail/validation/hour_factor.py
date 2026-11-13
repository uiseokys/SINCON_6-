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

K_, A_, WO, WH = 12, 5.0, 0.15, 0.3
tr = pd.read_csv(rs.B / "data/train.csv", parse_dates=["date"]); tr["y_log"] = np.log1p(tr.boardings)
LH = tr.pivot(index="date", columns=["station_id", "hour"], values="y_log"); AH = roll_anom(LH)
res = {}
for line in open("../nnexp/out4/folds.txt"):
    end, vs, ve = line.split(); origin = pd.Timestamp(end); m = A[A.origin.eq(origin)].reset_index(drop=True)
    y = m.y_log.to_numpy(); p = m[hb.MODEL].to_numpy(); hol = m.date.isin(days).to_numpy()
    lun = (m.station_id.eq(3122) & m.date.isin(pd.to_datetime(["2025-10-05", "2025-10-06", "2026-02-16", "2026-02-17"]))).to_numpy(); free = m.free.to_numpy() & ~lun
    dts = pd.DatetimeIndex(sorted(m.date.unique())); sb = rs.dow_base(LV, origin).stack().rename("b").reset_index(); sb.columns = ["dow", "station_id", "b"]
    g = [m.date.to_numpy(), m.station_id.to_numpy()]
    lvl = m.groupby(["date", "station_id"])[hb.MODEL].transform("mean").to_numpy(); b = m[["date", "station_id"]].assign(dow=m.date.dt.dayofweek).merge(sb, on=["dow", "station_id"], how="left").b.to_numpy(); ahat = lvl - b
    q = df.loc[df["date"].between(vs, ve), ["date", "station_id", "hour"]].reset_index(drop=True); q["date"] = pd.to_datetime(q["date"]); q["dn"] = np.load(f"../nnexp/out4/xgbnb_{end}.npy") - np.load(f"../nnexp/out4/xgb_{end}.npy")
    dn = np.where(free, m.merge(q, on=hb.KEY, how="left").dn.to_numpy(), 0.0); lv = pd.Series(dn).groupby(g).transform("mean").to_numpy(); hl = np.where(hol, np.clip(0.5 * lv, -CLIP, CLIP), 0.0)
    Z = (OB.reindex(dts) - rs.dow_base(OB, origin).reindex(dts.dayofweek).to_numpy()).to_numpy()
    d = AOB.index[(AOB.index <= origin) & AOB.notna().all(axis=1).to_numpy()]; X = AOB.loc[d].to_numpy(); mu = X.mean(0); U, s_, Vt = np.linalg.svd(X - mu, full_matrices=False)
    V = Vt[:K_].T; F = (X - mu) @ V; sd = F.std(0); F = F / sd; Ft = pd.DataFrame(((Z - mu) @ V) / sd, index=dts)
    def coefs(Y):
        out = {}
        for col in Y.columns:
            yv = Y[col].to_numpy(); ok = np.isfinite(yv)
            if ok.sum() >= 60: out[col] = np.linalg.solve(F[ok].T @ F[ok] + A_ * np.eye(K_), F[ok].T @ yv[ok])
        return out
    Bs = coefs(AL.reindex(d)); Bh = coefs(AH.reindex(d))
    Fm = Ft.reindex(m.date.to_numpy()).to_numpy()
    chat = (Fm * np.stack([Bs.get(s, np.full(K_, np.nan)) for s in m.station_id.to_numpy()])).sum(1)
    chh = (Fm * np.stack([Bh.get(k, np.full(K_, np.nan)) for k in zip(m.station_id.to_numpy(), m.hour.to_numpy())])).sum(1)
    c2 = np.where(free & np.isfinite(chat - ahat), chat - ahat, 0.0); cur = p + np.clip(np.where(hol, WH, WO) * c2, -CLIP, CLIP) + hl; rc = rm(cur, y)
    shp = chh - pd.Series(chh).groupby(g).transform("mean").to_numpy()                      # expected shape change (zero mean within the station-day)
    hbase = rs.dow_base(LH, origin).stack(level=[0, 1]).rename("b").reset_index(); hbase.columns = ["dow", "station_id", "hour", "b"]
    bh = m[["date", "station_id", "hour"]].assign(dow=m.date.dt.dayofweek).merge(hbase, on=["dow", "station_id", "hour"], how="left").b.to_numpy()
    ps = (p - bh); ps = ps - pd.Series(ps).groupby(g).transform("mean").to_numpy()            # our forecast's own shape change since the origin
    okk = free & np.isfinite(shp) & np.isfinite(ps); r = {}
    for w in (0.1, 0.2, 0.3):
        for scope, msk in (("all", okk), ("hol", okk & hol)):
            r[f"(ga) transfer {scope} w{w}"] = rm(cur + np.where(msk, np.clip(w * (shp - ps), -0.1, 0.1), 0.0), y) - rc
            r[f"(na) add {scope} w{w}"] = rm(cur + np.where(msk, np.clip(w * shp, -0.1, 0.1), 0.0), y) - rc
    r["corr(shape error, expected shape change) hol"] = float(np.corrcoef(((y - cur) - pd.Series(y - cur).groupby(g).transform("mean").to_numpy())[okk & hol], shp[okk & hol])[0, 1])
    res[end] = r
T = pd.DataFrame(res).T; cols = [c for c in T.columns if c.startswith("(")]; pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40)
print((T[cols] * 1e4).round(2).T.to_string()); print("mean", (T[cols].mean() * 1e4).round(2).to_dict()); print("worse", (T[cols] > 1e-9).sum().to_dict()); print(T.iloc[:, -1].round(3).to_dict())
json.dump(res, open("hour_factor.json", "w"), indent=1)
