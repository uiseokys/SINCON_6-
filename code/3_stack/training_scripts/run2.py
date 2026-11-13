"""팀 MLP 변형(PLE, Periodic, 잔차 학습) A/B/C 비교. 2026-06-30 이전 자료만. 기기: MPS."""
import sys, time, pickle, json, math
import numpy as np, torch, torch.nn as nn
data, variant, fold, device, out = sys.argv[1:6]
D = pickle.load(open(data, "rb")); df, FEATURES = D["df"], D["features"]
FOLDS = {"A": ("2026-02-28", "2026-03-01", "2026-06-30"), "B": ("2025-12-31", "2026-01-01", "2026-04-30"), "C": ("2025-10-31", "2025-11-01", "2026-02-28")}
EMBED_COLS = ("station_id", "hour", "dow", "month", "holiday_kind"); N_HOURS, FIRST_HOUR = 18, 6
T_BINS, K_FREQ, D_EMB, SIGMA = 16, 8, 8, 1.0


def categorical(d, si):
    st = d["station_id"].map(si).to_numpy(); hour = (d["hour"] - FIRST_HOUR).to_numpy(); dow = d["dow"].to_numpy(); rel = d["lunar_rel_day"].to_numpy(); n = len(si)
    cols = [(st, n, 24), (hour, N_HOURS, 8), (dow, 7, 4), ((d["month"] - 1).to_numpy(), 12, 4), (d["holiday_kind"].to_numpy(), 4, 3),
            (st * N_HOURS + hour, n * N_HOURS, 24), (st * 7 + dow, n * 7, 12), (np.where(np.isnan(rel), 11, rel + 5).astype(int), 12, 4)]
    return np.stack([c[0] for c in cols], 1).astype(np.int64), [(n_, dm) for _, n_, dm in cols]


class NumEmbed(nn.Module):
    """수치형 피처 임베딩. kind='ple' 또는 'periodic'. 입력 (B, F) → 출력 (B, F*D_EMB)."""
    def __init__(s, kind, x_train):
        super().__init__(); s.kind = kind; F = x_train.shape[1]
        if kind == "ple":
            lo = np.zeros((F, T_BINS), np.float32); wd = np.ones((F, T_BINS), np.float32); ok = np.zeros((F, T_BINS), np.float32)
            for f in range(F):
                e = np.unique(np.quantile(x_train[:, f], np.linspace(0, 1, T_BINS + 1)))
                if len(e) < 2: continue
                k = len(e) - 1; lo[f, :k] = e[:-1]; wd[f, :k] = np.maximum(e[1:] - e[:-1], 1e-6); ok[f, :k] = 1
            s.register_buffer("lo", torch.tensor(lo)); s.register_buffer("wd", torch.tensor(wd)); s.register_buffer("ok", torch.tensor(ok)); width = T_BINS
        else:
            s.freq = nn.Parameter(torch.randn(F, K_FREQ) * SIGMA); width = 2 * K_FREQ
        s.w = nn.Parameter(torch.randn(F, width, D_EMB) / math.sqrt(width)); s.b = nn.Parameter(torch.zeros(F, D_EMB))
    def forward(s, x):
        if s.kind == "ple":
            z = (((x[:, :, None] - s.lo) / s.wd).clamp(0, 1)) * s.ok
        else:
            v = 2 * math.pi * s.freq * x[:, :, None]; z = torch.cat([torch.sin(v), torch.cos(v)], -1)
        return torch.relu(torch.einsum("bft,ftd->bfd", z, s.w) + s.b).flatten(1)


class Net(nn.Module):
    def __init__(s, emb, n_num, n_real, width, drop, kind, x_train):
        super().__init__(); s.emb = nn.ModuleList([nn.Embedding(n, d) for n, d in emb]); s.n_real = n_real
        s.num = NumEmbed(kind, x_train[:, :n_real]) if kind else None
        prev = n_num + sum(d for _, d in emb) + (n_real * D_EMB if kind else 0); L = []
        for w in (width, width, width // 2):
            L += [nn.Linear(prev, w), nn.SiLU(), nn.Dropout(drop)]; prev = w
        s.mlp = nn.Sequential(*L); s.out = nn.Linear(prev, 1)
    def forward(s, xc, xn):
        parts = [e(xc[:, i]) for i, e in enumerate(s.emb)] + [xn]
        if s.num is not None: parts.append(s.num(xn[:, :s.n_real]))
        return s.out(s.mlp(torch.cat(parts, 1))).squeeze(1)


end, vs, ve = FOLDS[fold]
trm = (df["date"] <= end).to_numpy(); tem = df["date"].between(vs, ve).to_numpy()
train, target = df[trm], df[tem]
epochs, width, drop, lr, wd, bs = 30, 512, 0.15, 2e-3, 1e-5, 2048
si = {s: i for i, s in enumerate(sorted(train["station_id"].unique()))}
num = [f for f in FEATURES if f not in EMBED_COLS]
xc_tr, emb = categorical(train, si); xc_te, _ = categorical(target, si)
def numeric(d, nc=None):
    x = d[num].to_numpy(np.float32, copy=True); m = np.isnan(x)
    if nc is None: nc = [i for i in range(x.shape[1]) if m[:, i].any()]
    return np.nan_to_num(x, nan=0.0), m[:, nc].astype(np.float32), nc
xr_tr, mk_tr, nc = numeric(train); xr_te, mk_te, _ = numeric(target, nc)
y = np.log1p(train["boardings"].to_numpy(np.float32)); yv = np.log1p(target["boardings"].to_numpy(np.float64))
offset_tr = offset_te = None
if variant == "resid":
    offset_tr = np.load(f"{out}/lgb_{fold}_trainoof.npy").astype(np.float32); offset_te = np.load(f"{out}/lgb_{fold}_test.npy").astype(np.float32)
    xr_tr = np.concatenate([xr_tr, offset_tr[:, None]], 1); xr_te = np.concatenate([xr_te, offset_te[:, None]], 1)
n_real = xr_tr.shape[1]
xr_te = np.clip(xr_te, xr_tr.min(0), xr_tr.max(0)); mu, sd = xr_tr.mean(0), xr_tr.std(0); sd = np.where(sd < 1e-3, 1.0, sd)
xn_tr = np.concatenate([(xr_tr - mu) / sd, mk_tr], 1); xn_te = np.concatenate([(xr_te - mu) / sd, mk_te], 1)
t_raw = y - offset_tr if variant == "resid" else y
ym, ys = float(t_raw.mean()), float(t_raw.std())
kind = {"base": None, "resid": None, "ple": "ple", "periodic": "periodic"}[variant]
dev = torch.device(device)
xc_t, xn_t, y_t = torch.tensor(xc_tr, device=dev), torch.tensor(xn_tr, device=dev), torch.tensor((t_raw - ym) / ys, device=dev)
xc_v, xn_v = torch.tensor(xc_te, device=dev), torch.tensor(xn_te, device=dev)
n = len(y); steps = epochs * ((n + bs - 1) // bs); res = {}
for seed in [int(s) for s in (sys.argv[6].split(",") if len(sys.argv) > 6 else ["0", "1"])]:
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
    if variant == "resid": p = p + offset_te
    np.save(f"{out}/{variant}_{fold}_s{seed}.npy", p)
    res[seed] = (float(np.sqrt(np.mean((p - yv) ** 2))), round(time.time() - t0, 1))
    print(variant, fold, seed, res[seed], flush=True)
p2 = np.mean([np.load(f"{out}/{variant}_{fold}_s{s}.npy") for s in res], 0)
json.dump({"variant": variant, "fold": fold, "seeds": res, "rmse_2seed": float(np.sqrt(np.mean((p2 - yv) ** 2)))}, open(f"{out}/{variant}_{fold}.json", "w"))
