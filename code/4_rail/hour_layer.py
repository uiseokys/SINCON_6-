"""시간대별 반응 층: 다른 노선 주성분으로 본 '예상 형태 변화'와 우리 예측의 형태 변화의 차이를 0.1만큼 옮긴다."""
import numpy as np, pandas as pd
import factor_layer as fl
W, CLIP, K, ALPHA = 0.1, 0.1, 12, 5.0
def apply(test, pred, shape_pred, train, rail_all, blocked_station_days, origin=fl.ORIGIN):
    """pred: 이 층을 더할 예측. shape_pred: 형태를 재는 예측(수준 층을 더하기 전과 형태가 같으므로 pred 를 넣어도 된다)."""
    assert train["date"].max() <= origin and not rail_all["line"].eq("7호선").any()
    ob = fl.other_matrix(rail_all); ao = fl.roll_anom(ob)
    lh = np.log1p(train.pivot(index="date", columns=["station_id", "hour"], values="boardings")); ah = fl.roll_anom(lh)
    d = ao.index[(ao.index <= origin) & ao.notna().all(axis=1).to_numpy()]; X = ao.loc[d].to_numpy(); mu = X.mean(0); _, s, vt = np.linalg.svd(X - mu, full_matrices=False)
    V = vt[:K].T; F = (X - mu) @ V; sd = F.std(0); F = F / sd; Y = ah.reindex(d); coef = {}
    for col in Y.columns:
        yv = Y[col].to_numpy(); ok = np.isfinite(yv)
        if ok.sum() >= 60: coef[col] = np.linalg.solve(F[ok].T @ F[ok] + ALPHA * np.eye(K), F[ok].T @ yv[ok])
    dts = pd.DatetimeIndex(sorted(test["date"].unique())); Z = ob.reindex(dts) - fl.dow_base(ob, origin).reindex(dts.dayofweek).to_numpy()
    Ft = pd.DataFrame(((Z.to_numpy() - mu) @ V) / sd, index=dts).reindex(test["date"].to_numpy()).to_numpy()
    Bm = np.stack([coef.get(k, np.full(K, np.nan)) for k in zip(test["station_id"].to_numpy(), test["hour"].to_numpy())]); chh = (Ft * Bm).sum(1)
    g = [test["date"].to_numpy(), test["station_id"].to_numpy()]; shp = chh - pd.Series(chh).groupby(g).transform("mean").to_numpy()
    hb_ = fl.dow_base(lh, origin).stack(level=[0, 1]).rename("b").reset_index(); hb_.columns = ["dow", "station_id", "hour", "b"]
    bh = test[["date", "station_id", "hour"]].assign(dow=test["date"].dt.dayofweek).merge(hb_, on=["dow", "station_id", "hour"], how="left")["b"].to_numpy()
    ps = np.log1p(np.asarray(shape_pred, float)) - bh; ps = ps - pd.Series(ps).groupby(g).transform("mean").to_numpy()
    free = ~pd.MultiIndex.from_frame(test[["date", "station_id"]]).isin(blocked_station_days) & np.isfinite(shp) & np.isfinite(ps)
    add = np.where(free, np.clip(W * (np.nan_to_num(shp) - np.nan_to_num(ps)), -CLIP, CLIP), 0.0)
    lp = np.log1p(np.asarray(pred, float)); out = np.asarray(pred, float).copy(); ch = add != 0; out[ch] = np.expm1(np.maximum(lp[ch] + add[ch], 0.0))
    by_hour = pd.Series(add[free]).groupby(test["hour"].to_numpy()[free]).mean().round(4).to_dict()
    return out, {"changed_rows": int(ch.sum()), "max_abs_log_change": float(np.abs(add).max()), "rms_log_change": float(np.sqrt((add ** 2).mean())), "station_hour_coefficients": len(coef), "mean_change_by_hour": {int(k): float(v) for k, v in by_hour.items()},
                 "max_station_day_mean_change": float(pd.Series(add).groupby(g).mean().abs().max())}
