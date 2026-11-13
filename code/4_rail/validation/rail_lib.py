import numpy as np, pandas as pd
GROUPS = {"G1": ("경인선", ["백운", "동암", "간석", "부개"]), "G2": ("경인선", ["송내", "중동", "부천", "소사", "역곡"]), "G3": ("경인선", ["도화", "제물포", "도원", "동인천", "인천"]),
          "G4": ("수인선", ["송도", "연수", "남동인더스파크", "호구포", "인천논현", "소래포구", "월곶"]), "G5": ("수인선", ["인천", "신포", "숭의", "인하대"])}
SETS = {"both": ["G1", "G2", "G3", "G4", "G5"], "gyeongin": ["G1", "G2", "G3"], "suin": ["G4", "G5"]}
def signals(path):
    """date x 10 signals: group mean of (log1p value − median of the same weekday over the previous 8 weeks)."""
    d = pd.read_csv(path, parse_dates=["date"]); out = {}
    for col in ("board", "alight"):
        w = np.log1p(d.pivot(index="date", columns=["line", "station"], values=col)).sort_index()
        base = pd.concat([w.shift(7 * k) for k in range(1, 9)], keys=range(8)).groupby(level=1).median()
        a = w - base
        for g, (line, sts) in GROUPS.items(): out[f"{g}_{col}"] = a[[(line, s) for s in sts]].mean(axis=1)
    return pd.DataFrame(out).dropna()
def cols(which): return [f"{g}_{c}" for g in SETS[which] for c in ("board", "alight")]
def fit(prior, Z, which, a1=10.0, a2=50.0):
    """prior: station-day residuals (date, station_id, r). Returns global and per-station coefficients on standardized signals."""
    c = cols(which); p = prior.merge(Z[c], left_on="date", right_index=True, how="inner")
    mu, sd = Z.loc[Z.index.isin(p.date.unique()), c].mean(), Z.loc[Z.index.isin(p.date.unique()), c].std().replace(0, 1.0)
    X = ((p[c] - mu) / sd).to_numpy(); rc = (p.r - p.groupby("station_id").r.transform("mean")).to_numpy()
    k = len(c); bg = np.linalg.solve(X.T @ X + a1 * np.eye(k), X.T @ rc); rest = rc - X @ bg; bs = {}
    for s, idx in p.groupby("station_id").indices.items():
        Xs = X[idx]; bs[s] = np.linalg.solve(Xs.T @ Xs + a2 * np.eye(k), Xs.T @ rest[idx])
    return dict(cols=c, mu=mu, sd=sd, bg=bg, bs=bs, n=len(p), days=int(p.date.nunique()))
def predict(model, keys, Z, per_station=True):
    """keys: frame with date, station_id. Returns predicted station-day residual (0 where signals are missing)."""
    z = ((Z[model["cols"]] - model["mu"]) / model["sd"]).reindex(keys.date.to_numpy()).to_numpy(); ok = np.isfinite(z).all(1); z = np.nan_to_num(z)
    out = z @ model["bg"]
    if per_station:
        B = np.stack([model["bs"].get(s, np.zeros(len(model["cols"]))) for s in keys.station_id.to_numpy()]); out = out + (z * B).sum(1)
    return np.where(ok, out, 0.0)
