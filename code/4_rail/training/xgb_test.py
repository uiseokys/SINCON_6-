"""전체 학습(<= 2026-06-30) → 시험 기간 XGBoost·LightGBM 예측. 설정은 9개 기준일 검증과 같다."""
import pickle, numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb
D = pickle.load(open("data.pkl", "rb")); tr, FEATURES = D["df"], D["features"]; te = pickle.load(open("test.pkl", "rb"))["df"]
assert str(tr["date"].max())[:10] == "2026-06-30" and len(te) == 157194
X = pd.concat([tr[FEATURES], te[FEATURES]], ignore_index=True)
for c in [c for c in X.columns if X[c].dtype == object] + ["station_id"]: X[c] = X[c].astype("category")
n = len(tr); y = np.log1p(tr["boardings"].to_numpy(np.float64))
m = xgb.XGBRegressor(tree_method="hist", max_depth=8, learning_rate=0.05, n_estimators=800, subsample=0.8, colsample_bytree=0.8, enable_categorical=True, n_jobs=8, random_state=0)
m.fit(X.iloc[:n], y); np.save("out3/xgb_test.npy", m.predict(X.iloc[n:]))
PL = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=0, verbose=-1, num_threads=8)
np.save("out3/lgb_test.npy", lgb.train(PL, lgb.Dataset(X.iloc[:n], y), 600).predict(X.iloc[n:]))
print("done", np.load("out3/xgb_test.npy").mean().round(4), np.load("out3/lgb_test.npy").mean().round(4))
