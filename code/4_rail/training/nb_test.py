import pickle, numpy as np, pandas as pd, xgboost as xgb
D = pickle.load(open("data.pkl", "rb")); tr, FEATURES = D["df"], D["features"]; te = pickle.load(open("test.pkl", "rb"))["df"]
NB = pd.read_pickle("out4/nb_features.pkl"); NBC = list(NB.columns)
X = pd.concat([tr[FEATURES], te[FEATURES]], ignore_index=True); dts = pd.to_datetime(pd.concat([tr["date"], te["date"]], ignore_index=True)); nbv = NB.reindex(dts.to_numpy()).to_numpy(); assert np.isfinite(nbv[len(tr):]).all()
for i, c in enumerate(NBC): X[c] = nbv[:, i]
for c in [c for c in X.columns if X[c].dtype == object] + ["station_id"]: X[c] = X[c].astype("category")
n = len(tr); y = np.log1p(tr["boardings"].to_numpy(np.float64))
m = xgb.XGBRegressor(tree_method="hist", max_depth=8, learning_rate=0.05, n_estimators=800, subsample=0.8, colsample_bytree=0.8, enable_categorical=True, n_jobs=8, random_state=0)
m.fit(X.iloc[:n], y); p = m.predict(X.iloc[n:]); np.save("out3/xgbnb_test.npy", p); print("xgb+nb test mean", p.mean().round(4), "| plain", np.load("out3/xgb_test.npy").mean().round(4))
