"""임베딩 MLP (PyTorch): LightGBM과 구조가 다른 두 번째 모델. GPU가 있으면 GPU로 학습합니다.

장치 선택: CUDA(Kaggle·Colab GPU) -> MPS(Apple Silicon GPU) -> CPU 순으로 자동 선택.
입력
  - 범주형(임베딩): 역, 시간, 요일, 월, 공휴일 종류, 역x시간, 역x요일, 명절 상대 일수
  - 수치형: 나머지 특징. 학습 구간의 최소·최대로 잘라(외삽 방지) 표준화하고, NaN은 0 + 결측 표시 열로 처리
타깃: log1p(승차)를 표준화해 MSE로 학습 (= RMSLE 기준). 예측은 seed 여러 개의 평균.
주의: GPU 연산 특성상 장치가 다르면 예측값이 소수점 아래에서 조금 달라질 수 있습니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DEVICE = ("cuda" if __import__("os").environ.get("SINCON_DEVICE", "auto") != "cpu" and torch.cuda.is_available() else "cpu")
EMBED_COLS = ("station_id", "hour", "dow", "month", "holiday_kind")   # 수치 입력에서 빼고 임베딩으로만 쓰는 열
N_HOURS, FIRST_HOUR = 18, 6


def _categorical(df: pd.DataFrame, station_index: dict) -> tuple[np.ndarray, list]:
    """임베딩 입력 (행 x 8)과 각 열의 (범주 수, 임베딩 차원)."""
    st = df["station_id"].map(station_index).to_numpy()
    hour = (df["hour"] - FIRST_HOUR).to_numpy()
    dow = df["dow"].to_numpy()
    rel = df["lunar_rel_day"].to_numpy()
    n_st = len(station_index)
    cols = [
        (st, n_st, 24), (hour, N_HOURS, 8), (dow, 7, 4), ((df["month"] - 1).to_numpy(), 12, 4),
        (df["holiday_kind"].to_numpy(), 4, 3),
        (st * N_HOURS + hour, n_st * N_HOURS, 24), (st * 7 + dow, n_st * 7, 12),
        (np.where(np.isnan(rel), 11, rel + 5).astype(int), 12, 4),
    ]
    return np.stack([c[0] for c in cols], axis=1).astype(np.int64), [(n, dim) for _, n, dim in cols]


class _Net(nn.Module):
    def __init__(self, emb_sizes, n_num, width, drop):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(n, dim) for n, dim in emb_sizes])
        d_in = n_num + sum(dim for _, dim in emb_sizes)
        layers, prev = [], d_in
        for w in (width, width, width // 2):
            layers += [nn.Linear(prev, w), nn.SiLU(), nn.Dropout(drop)]
            prev = w
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(prev, 1)

    def forward(self, xc, xn):
        return self.out(self.mlp(torch.cat([e(xc[:, i]) for i, e in enumerate(self.emb)] + [xn], dim=1))).squeeze(1)


def fit_predict_nn(train: pd.DataFrame, target: pd.DataFrame, features: list, seeds,
                   epochs: int = 30, batch_size: int = 2048, lr: float = 2e-3, weight_decay: float = 1e-5,
                   width: int = 512, drop: float = 0.05) -> np.ndarray:
    """train으로 학습해 target 행의 log1p(승차) 예측을 돌려준다 (seed 평균)."""
    station_index = {s: i for i, s in enumerate(sorted(train["station_id"].unique()))}
    num_cols = [f for f in features if f not in EMBED_COLS]
    xc_tr, emb_sizes = _categorical(train, station_index)
    xc_te, _ = _categorical(target, station_index)

    def numeric(df, nan_cols=None):
        x = df[num_cols].to_numpy(dtype=np.float32, copy=True)
        mask = np.isnan(x)
        if nan_cols is None:
            nan_cols = [i for i in range(x.shape[1]) if mask[:, i].any()]
        return np.concatenate([np.nan_to_num(x, nan=0.0), mask[:, nan_cols].astype(np.float32)], axis=1), nan_cols

    xn_tr, nan_cols = numeric(train)
    xn_te, _ = numeric(target, nan_cols)
    lo, hi = xn_tr.min(0), xn_tr.max(0)
    xn_te = np.clip(xn_te, lo, hi)                      # 학습에서 본 범위 밖으로 외삽하지 않음 (시간 축 포함)
    mu, sd = xn_tr.mean(0), xn_tr.std(0)
    sd = np.where(sd < 1e-3, 1.0, sd)                   # 학습 구간에서 값이 하나뿐인 열 보호
    xn_tr, xn_te = (xn_tr - mu) / sd, (xn_te - mu) / sd
    y = np.log1p(train["boardings"].to_numpy(dtype=np.float32))
    y_mean, y_std = float(y.mean()), float(y.std())

    xc_t = torch.tensor(xc_tr, device=DEVICE)
    xn_t = torch.tensor(xn_tr, device=DEVICE)
    y_t = torch.tensor((y - y_mean) / y_std, device=DEVICE)
    xc_v = torch.tensor(xc_te, device=DEVICE)
    xn_v = torch.tensor(xn_te, device=DEVICE)
    n = len(y)
    total_steps = epochs * ((n + batch_size - 1) // batch_size)
    pred = np.zeros(len(target), dtype=np.float64)
    for seed in seeds:
        print(f"MLP seed={seed}, epochs={epochs}, width={width}, device={DEVICE}", flush=True)
        torch.manual_seed(seed)
        net = _Net(emb_sizes, xn_tr.shape[1], width, drop).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total_steps, pct_start=0.1)
        for _ in range(epochs):                         # 에폭 수 고정 (검증 세트로 조기 종료하지 않음)
            net.train()
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad(set_to_none=True)
                loss = ((net(xc_t[idx], xn_t[idx]) - y_t[idx]) ** 2).mean()
                loss.backward()
                opt.step()
                sched.step()
        net.eval()
        with torch.no_grad():
            out = [net(xc_v[i:i + 65536], xn_v[i:i + 65536]).float().cpu().numpy() for i in range(0, len(target), 65536)]
        pred += (np.concatenate(out) * y_std + y_mean) / len(seeds)
    return pred
