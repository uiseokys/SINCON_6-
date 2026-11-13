"""전체 학습(<= 2026-06-30) 후 시험 기간 예측. 변형: base / ple / periodic. 기기: MPS."""
import sys, time, pickle, math
import numpy as np, torch, torch.nn as nn
data, testp, variant, device, out, seeds = sys.argv[1:7]
src = open("run2.py").read()
ns = {"np": np, "nn": nn, "torch": torch, "math": math, "N_HOURS": 18, "FIRST_HOUR": 6, "T_BINS": 16, "K_FREQ": 8, "D_EMB": 8, "SIGMA": 1.0}
exec(src[src.index("def categorical"):src.index("end, vs, ve = FOLDS[fold]")], ns)
categorical, Net = ns["categorical"], ns["Net"]
D = pickle.load(open(data, "rb")); train, FEATURES = D["df"], D["features"]; target = pickle.load(open(testp, "rb"))["df"]
EMBED_COLS = ("station_id", "hour", "dow", "month", "holiday_kind")
epochs, width, drop, lr, wd, bs = 30, 512, 0.15, 2e-3, 1e-5, 2048
si = {s: i for i, s in enumerate(sorted(train["station_id"].unique()))}
num = [f for f in FEATURES if f not in EMBED_COLS]
xc_tr, emb = categorical(train, si); xc_te, _ = categorical(target, si)
def numeric(d, nc=None):
    x = d[num].to_numpy(np.float32, copy=True); m = np.isnan(x)
    if nc is None: nc = [i for i in range(x.shape[1]) if m[:, i].any()]
    return np.nan_to_num(x, nan=0.0), m[:, nc].astype(np.float32), nc
xr_tr, mk_tr, nc = numeric(train); xr_te, mk_te, _ = numeric(target, nc); n_real = xr_tr.shape[1]
xr_te = np.clip(xr_te, xr_tr.min(0), xr_tr.max(0)); mu, sd = xr_tr.mean(0), xr_tr.std(0); sd = np.where(sd < 1e-3, 1.0, sd)
xn_tr = np.concatenate([(xr_tr - mu) / sd, mk_tr], 1); xn_te = np.concatenate([(xr_te - mu) / sd, mk_te], 1)
y = np.log1p(train["boardings"].to_numpy(np.float32)); ym, ys = float(y.mean()), float(y.std())
kind = {"base": None, "ple": "ple", "periodic": "periodic"}[variant]; dev = torch.device(device)
xc_t, xn_t, y_t = torch.tensor(xc_tr, device=dev), torch.tensor(xn_tr, device=dev), torch.tensor((y - ym) / ys, device=dev)
xc_v, xn_v = torch.tensor(xc_te, device=dev), torch.tensor(xn_te, device=dev)
n = len(y); steps = epochs * ((n + bs - 1) // bs)
for seed in [int(s) for s in seeds.split(",")]:
    t0 = time.time(); torch.manual_seed(seed)
    net = Net(emb, xn_tr.shape[1], n_real, width, drop, kind, xn_tr).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    for _ in range(epochs):
        net.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad(set_to_none=True)
            loss = ((net(xc_t[idx], xn_t[idx]) - y_t[idx]) ** 2).mean(); loss.backward(); opt.step(); sch.step()
    net.eval()
    with torch.no_grad():
        p = np.concatenate([net(xc_v[i:i + 65536], xn_v[i:i + 65536]).float().cpu().numpy() for i in range(0, len(target), 65536)]) * ys + ym
    np.save(f"{out}/{variant}_test_s{seed}.npy", p)
    print(variant, "test seed", seed, "mean log pred", round(float(p.mean()), 4), "sec", round(time.time() - t0, 1), flush=True)
