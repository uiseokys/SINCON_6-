"""추가 모델 검증: XGBoost, CatBoost, DLinear, NLinear (검증 A/B/C, 각 기준일까지의 자료만 사용).

  XGBoost·CatBoost : ensemble.py와 같은 특징 68개로 log1p(승차)를 학습
  DLinear·NLinear  : 역x시간 계열(1,278개)의 최근 L일 log1p 승차만 입력으로, 이후 H일을 한 번에 예측하는
                     선형 시계열 모델 (달력·날씨·행사 정보는 쓰지 않음, 모든 계열이 가중치를 공유)

실행 (sincon-baseline 폴더에서)
  python experiments/model_zoo.py --models xgb,dlinear,nlinear            # 수 분
  python experiments/model_zoo.py --models catboost                       # 구간당 약 5분
출력: experiments/valid_predictions_zoo.csv.gz (fold, date, station_id, hour, y_log, 모델별 log 예측)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ensemble as en                     # noqa: E402
import improved as im                     # noqa: E402
from make_calendar import build_calendar  # noqa: E402

OUT = HERE / "valid_predictions_zoo.csv.gz"
KEYS = ["fold", "date", "station_id", "hour"]


def rmse(a, b):
    return float(np.sqrt(np.mean(np.square(a - b))))


# ------------------------------------------------------------------ tree models
def fit_xgb(a, b, seeds=(0,), rounds=2000):
    import xgboost as xgb
    cat = pd.CategoricalDtype(sorted(a["station_id"].unique()))
    def frame(d):
        x = d[en.FEATURES].copy()
        x["station_id"] = x["station_id"].astype(cat)
        return x
    dtr = xgb.DMatrix(frame(a), np.log1p(a["boardings"].to_numpy()), enable_categorical=True)
    dva = xgb.DMatrix(frame(b), enable_categorical=True)
    pred = np.zeros(len(b))
    for seed in seeds:
        params = dict(objective="reg:squarederror", tree_method="hist", eta=0.05, max_depth=8, min_child_weight=20,
                      subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, max_cat_to_onehot=1, seed=seed, nthread=16)
        pred += xgb.train(params, dtr, rounds).predict(dva) / len(seeds)
    return pred


def fit_catboost(a, b, iterations=8000):
    from catboost import CatBoostRegressor, Pool
    model = CatBoostRegressor(loss_function="RMSE", depth=8, learning_rate=0.15, iterations=iterations, random_seed=0,
                              thread_count=14, verbose=0)
    model.fit(Pool(a[en.FEATURES], np.log1p(a["boardings"].to_numpy()), cat_features=["station_id"]))
    return model.predict(b[en.FEATURES])


# ------------------------------------------------------------------ linear time-series models
def fit_linear_ts(kind, a, b, lookback=112, seeds=(0, 1, 2), epochs=40, kernel=25):
    """DLinear / NLinear on station x hour daily series. Returns log predictions aligned with rows of b."""
    import torch
    import torch.nn as nn
    dev = ("cuda" if __import__("os").environ.get("SINCON_DEVICE", "auto") != "cpu" and torch.cuda.is_available() else "cpu")
    wide = a.assign(v=np.log1p(a["boardings"])).pivot_table(index=["station_id", "hour"], columns="date", values="v")
    days = pd.date_range(wide.columns.min(), wide.columns.max()).strftime("%Y-%m-%d")
    wide = wide.reindex(columns=days)                       # NaN before a station opened
    target_days = sorted(b["date"].unique())
    horizon = len(target_days)
    assert pd.Timestamp(target_days[0]) == pd.Timestamp(days[-1]) + pd.Timedelta(days=1)
    x_all = torch.tensor(wide.to_numpy(dtype=np.float32))
    win = x_all.unfold(1, lookback + horizon, 1).reshape(-1, lookback + horizon)
    win = win[~torch.isnan(win).any(dim=1)].to(dev)         # training windows lie entirely before the cutoff
    last = x_all[:, -lookback:]
    usable = ~torch.isnan(last).any(dim=1)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(lookback, horizon)
            if kind == "dlinear":
                self.b = nn.Linear(lookback, horizon)
                for lin in (self.a, self.b):
                    nn.init.constant_(lin.weight, 1.0 / lookback)
                    nn.init.zeros_(lin.bias)
        def forward(self, x):
            if kind == "nlinear":                            # subtract the last value, predict, add it back
                anchor = x[:, -1:]
                return self.a(x - anchor) + anchor
            pad = (kernel - 1) // 2                          # DLinear: moving-average trend + remainder
            padded = torch.cat([x[:, :1].repeat(1, pad), x, x[:, -1:].repeat(1, pad)], dim=1)
            trend = padded.unfold(1, kernel, 1).mean(dim=2)
            return self.a(x - trend) + self.b(trend)

    out = np.zeros((len(wide), horizon))
    for seed in seeds:
        torch.manual_seed(seed)
        net = Model().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        n = len(win)
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, 1024):
                w = win[perm[i:i + 1024]]
                opt.zero_grad(set_to_none=True)
                loss = ((net(w[:, :lookback]) - w[:, lookback:]) ** 2).mean()
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            p = net(torch.nan_to_num(last).to(dev)).cpu().numpy()
        out += p / len(seeds)
    out[~usable.numpy()] = np.nan                            # series without a full lookback window
    pred = pd.DataFrame(out, index=wide.index, columns=target_days).stack().rename("p")
    pred.index.names = ["station_id", "hour", "date"]
    got = b[["station_id", "hour", "date"]].join(pred, on=["station_id", "hour", "date"])["p"].to_numpy()
    return got, int(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="xgb,dlinear,nlinear")
    args = ap.parse_args()
    wanted = args.models.split(",")
    meta = pd.read_csv(HERE.parent / "data" / "metaData.csv")[["station_id", "canonical_line"]].rename(columns={"canonical_line": "line"})
    tr = im.enrich(pd.read_csv(HERE.parent / "data" / "train.csv"), build_calendar(), meta, True)
    result = pd.read_csv(OUT) if OUT.exists() else None
    parts = []
    for name, train_end, start, end in en.VALID_FOLDS:
        a, b = tr[tr["date"] <= train_end], tr[tr["date"].between(start, end)]
        y = np.log1p(b["boardings"].to_numpy())
        fold = b[["date", "station_id", "hour"]].reset_index(drop=True).assign(fold=name, y_log=y)[KEYS + ["y_log"]]
        for model in wanted:
            t = time.time()
            if model == "xgb":
                fold["xgb"] = fit_xgb(a, b)
            elif model == "catboost":
                fold["catboost"] = fit_catboost(a, b)
            elif model in ("dlinear", "nlinear"):
                fold[model], n = fit_linear_ts(model, a, b)
            covered = fold[model].notna()
            print(f"fold {name} {model}: RMSLE {rmse(fold.loc[covered, model], fold.loc[covered, 'y_log']):.5f} "
                  f"(예측 가능한 행 {covered.mean():.1%}) | {time.time() - t:.0f}초", flush=True)
        parts.append(fold)
    new = pd.concat(parts, ignore_index=True)
    if result is not None:                                   # keep columns from earlier runs
        assert (result[KEYS].astype(str).values == new[KEYS].astype(str).values).all()
        for c in result.columns:
            if c not in new.columns:
                new[c] = result[c].to_numpy()
    new.to_csv(OUT, index=False, float_format="%.6f")
    print("저장:", OUT.name, new.shape)


if __name__ == "__main__":
    main()
