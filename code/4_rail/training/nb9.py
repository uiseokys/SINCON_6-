import pickle, time, numpy as np, pandas as pd, xgboost as xgb
D = pickle.load(open("data.pkl", "rb")); df, FEATURES = D["df"], D["features"]
rail = pd.read_csv("../rail/rail_allowed.csv.gz", parse_dates=["date"])
def feats(rail):
    out = {}
    for name, m in (("gy", rail.line.eq("경인선")), ("su", rail.line.eq("수인선")), ("all", pd.Series(True, index=rail.index))):
        s = np.log(rail[m].groupby("date").board.sum()); out[f"nb_{name}_log"] = s
        out[f"nb_{name}_r28"] = s - pd.concat([s.shift(7 * k) for k in range(1, 5)], axis=1).median(axis=1)
    a = np.log(rail.groupby("date").alight.sum()); out["nb_all_alight_r28"] = a - pd.concat([a.shift(7 * k) for k in range(1, 5)], axis=1).median(axis=1)
    return pd.DataFrame(out)
NB = feats(rail); NB.to_pickle("out4/nb_features.pkl"); NBC = list(NB.columns); print("nb features", NB.shape, NB.index.min().date(), "~", NB.index.max().date(), NBC)
X = df[FEATURES].copy(); nbv = NB.reindex(pd.to_datetime(df["date"]).to_numpy()).to_numpy()
for i, c in enumerate(NBC): X[c] = nbv[:, i]
for c in [c for c in X.columns if X[c].dtype == object] + ["station_id"]: X[c] = X[c].astype("category")
y = np.log1p(df["boardings"].to_numpy(np.float64)); rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
for line in open("out4/folds.txt"):
    end, vs, ve = line.split(); t0 = time.time(); tr = (df["date"] <= end).to_numpy(); te = df["date"].between(vs, ve).to_numpy()
    m = xgb.XGBRegressor(tree_method="hist", max_depth=8, learning_rate=0.05, n_estimators=800, subsample=0.8, colsample_bytree=0.8, enable_categorical=True, n_jobs=8, random_state=0)
    m.fit(X[tr], y[tr]); p = m.predict(X[te]); np.save(f"out4/xgbnb_{end}.npy", p)
    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False); rk = {c: int(list(imp.index).index(c)) + 1 for c in NBC}
    print(end, "xgb+nb", round(rm(p, y[te]), 5), "| plain xgb", round(rm(np.load(f"out4/xgb_{end}.npy"), y[te]), 5), "| importance rank of nb features:", rk, f"| {time.time()-t0:.0f}s", flush=True)
