"""전 노선(7호선·환승역 제외) 일별 승차의 주성분으로 역별 예상 변화를 구해 옮기는 층. 값은 RAIL_PREREGISTRATION.md 에 고정."""
import numpy as np, pandas as pd
ORIGIN = pd.Timestamp("2026-06-30"); K, ALPHA, W_ORDINARY, W_HOLIDAY, CLIP, WEEKS = 5, 5.0, 0.1, 0.2, 0.15, 8
def roll_anom(frame): return frame - pd.concat([frame.shift(7 * k) for k in range(1, WEEKS + 1)], keys=range(WEEKS)).groupby(level=1).median()
def dow_base(frame, origin=ORIGIN):
    w = frame.loc[origin - pd.Timedelta(days=7 * WEEKS - 1):origin]; return w.groupby(w.index.dayofweek).median()
def other_matrix(rail_all):
    ob = np.log1p(rail_all.pivot_table(index="date", columns=["line", "station"], values="board", aggfunc="sum")).sort_index()
    return ob.loc[:, ob.notna().sum().eq(len(ob))]
def apply(test, pred, train, rail_all, special_days, blocked_station_days, origin=ORIGIN, K=K, W_ORDINARY=W_ORDINARY, W_HOLIDAY=W_HOLIDAY):
    assert train["date"].max() <= origin and not rail_all["line"].eq("7호선").any()
    ob = other_matrix(rail_all); lv = np.log1p(train.set_index(["date", "station_id", "hour"])["boardings"]).groupby(level=[0, 1]).mean().unstack()
    ao, al = roll_anom(ob), roll_anom(lv)
    d = ao.index[(ao.index <= origin) & ao.notna().all(axis=1).to_numpy()]; X = ao.loc[d].to_numpy(); mu = X.mean(0); _, s, vt = np.linalg.svd(X - mu, full_matrices=False)
    V = vt[:K].T; F = (X - mu) @ V; sd = F.std(0); F = F / sd; Y = al.reindex(d); coef = {}
    for sid in Y.columns:
        yv = Y[sid].to_numpy(); ok = np.isfinite(yv)
        if ok.sum() >= 60: coef[sid] = np.linalg.solve(F[ok].T @ F[ok] + ALPHA * np.eye(K), F[ok].T @ yv[ok])
    dts = pd.DatetimeIndex(sorted(test["date"].unique())); Z = ob.reindex(dts) - dow_base(ob, origin).reindex(dts.dayofweek).to_numpy(); assert Z.notna().all().all()
    Ft = pd.DataFrame(((Z.to_numpy() - mu) @ V) / sd, index=dts)
    Bm = np.stack([coef.get(s, np.full(K, np.nan)) for s in test["station_id"].to_numpy()]); chat = (Ft.reindex(test["date"].to_numpy()).to_numpy() * Bm).sum(1)
    base = dow_base(lv, origin).stack().rename("b").reset_index(); base.columns = ["dow", "station_id", "b"]
    lp = np.log1p(np.asarray(pred, float)); lvl = pd.Series(lp).groupby([test["date"].to_numpy(), test["station_id"].to_numpy()]).transform("mean").to_numpy()
    b = test[["date", "station_id"]].assign(dow=test["date"].dt.dayofweek).merge(base, on=["dow", "station_id"], how="left")["b"].to_numpy()
    c = chat - (lvl - b); hol = test["date"].isin(special_days).to_numpy(); free = ~pd.MultiIndex.from_frame(test[["date", "station_id"]]).isin(blocked_station_days) & np.isfinite(c)
    add = np.where(free, np.clip(np.where(hol, W_HOLIDAY, W_ORDINARY) * np.nan_to_num(c), -CLIP, CLIP), 0.0)
    out = np.asarray(pred, float).copy(); ch = add != 0; out[ch] = np.expm1(np.maximum(lp[ch] + add[ch], 0.0)); mon = test["date"].dt.month.to_numpy()
    info = {"other_stations": int(ob.shape[1]), "fit_days": int(len(d)), "variance_explained": float((s[:K] ** 2).sum() / (s ** 2).sum()), "stations_with_coefficients": len(coef), "changed_rows": int(ch.sum()), "blocked_rows": int((~free).sum()),
            "max_abs_log_change": float(np.abs(add).max()), "by_month_mean_change": {int(k): round(float(v), 5) for k, v in pd.Series(add).groupby(mon).mean().items()},
            "by_month_expected_change_from_other_lines": {int(k): round(float(v), 4) for k, v in pd.Series(chat).groupby(mon).mean().items()}, "by_month_our_forecast_change": {int(k): round(float(v), 4) for k, v in pd.Series(lvl - b).groupby(mon).mean().items()}}
    return out, info
