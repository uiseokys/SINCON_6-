"""공휴일류 날짜의 수준 보정: 타 노선 피처를 넣은 XGBoost와 넣지 않은 XGBoost의 차이(역·날짜 평균)에 0.5배."""
import numpy as np, pandas as pd
W, CLIP = 0.5, 0.15
def apply(test, pred, xgb_with, xgb_without, special_days, blocked_station_days):
    d = np.asarray(xgb_with, float) - np.asarray(xgb_without, float)
    lvl = pd.Series(d).groupby([test["date"].to_numpy(), test["station_id"].to_numpy()]).transform("mean").to_numpy()
    hol = test["date"].isin(special_days).to_numpy(); free = ~pd.MultiIndex.from_frame(test[["date", "station_id"]]).isin(blocked_station_days)
    add = np.where(hol & free, np.clip(W * lvl, -CLIP, CLIP), 0.0); lp = np.log1p(np.asarray(pred, float)); out = np.asarray(pred, float).copy(); ch = add != 0
    out[ch] = np.expm1(np.maximum(lp[ch] + add[ch], 0.0))
    by = pd.Series(add[ch]).groupby(test["date"].dt.strftime("%m-%d").to_numpy()[ch]).mean().round(4).to_dict()
    return out, {"changed_rows": int(ch.sum()), "max_abs_log_change": float(np.abs(add).max()), "mean_change_by_date": by}
