import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "../hb_pkg"); sys.path.insert(0, "."); import hourbias_layer as hb, rail_lib as rl
from pathlib import Path
A = pd.read_pickle("../nnexp/out4/all9.pkl"); B = Path("/Users/jeong-uiseok/sincon-2026/sincon-baseline"); P = Path("/Users/jeong-uiseok/.codex/.chatgpt-projects/g-p-6aac7228bb908191b5f1eb617f4ff9bd"); W = P / "final_package/payload/work"
days, events, protected = hb.read_protection(P / "hourbias_delivery/cache")
fest = pd.read_csv(W / "sincon_festival_station_days.csv", parse_dates=["date"]); ev = pd.read_csv(W / "external_inputs/external/events_daily.csv", parse_dates=["date"], encoding="utf-8-sig").dropna(subset=["station_id"])
extra = pd.MultiIndex.from_frame(pd.concat([fest[["date", "station_id"]], ev[["date", "station_id"]].astype({"station_id": int})]).drop_duplicates())
k = pd.MultiIndex.from_frame(A[hb.DAYKEY]); A["free"] = ~k.isin(events) & ~k.isin(extra)
rail = pd.read_csv("rail_allowed.csv.gz", parse_dates=["date"]); RB = np.log1p(rail.pivot(index="date", columns=["line", "station"], values="board")).sort_index()
tr = pd.read_csv(B / "data/train.csv", parse_dates=["date"]); tr["y_log"] = np.log1p(tr.boardings); LV = tr.groupby(["date", "station_id"]).y_log.mean().unstack()
SETS = {"all25": list(RB.columns), "gyeongin": [c for c in RB.columns if c[0] == "경인선"], "suin": [c for c in RB.columns if c[0] == "수인선"]}
rm = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2))); CLIP = 0.15; WS = (0.1, 0.2, 0.3, 0.5)
def dow_base(frame, origin):
    w = frame.loc[origin - pd.Timedelta(days=55):origin]; return w.groupby(w.index.dayofweek).median()       # rows: dow
def shift_terms(hold, origin):
    """per row: other-line change since origin (per set) and our forecast's own change since origin."""
    ob = dow_base(RB, origin); dts = pd.DatetimeIndex(sorted(hold.date.unique())); a = RB.reindex(dts) - ob.reindex(dts.dayofweek).to_numpy()
    abar = {name: a[cols].mean(axis=1) for name, cols in SETS.items()}
    sb = dow_base(LV, origin)                                           # dow x station (actuals up to the origin only)
    lvl = hold.groupby(["date", "station_id"])[hb.MODEL].transform("mean").to_numpy()
    base = sb.stack().rename("b").reset_index(); base.columns = ["dow", "station_id", "b"]
    b = hold[["date", "station_id"]].assign(dow=hold.date.dt.dayofweek).merge(base, on=["dow", "station_id"], how="left").b.to_numpy()
    ahat = lvl - b
    return {name: v.reindex(hold.date.to_numpy()).to_numpy() - ahat for name, v in abar.items()}
if __name__ == "__main__":
    res = {}
    for origin, hold in A.groupby("origin", sort=True):
        hold = hold.reset_index(drop=True); terms = shift_terms(hold, origin); y = hold.y_log.to_numpy(); p = hold[hb.MODEL].to_numpy(); free = hold.free.to_numpy(); r0 = rm(p, y); r = {}
        hol = hold.date.isin(days).to_numpy(); lead = (hold.date - origin).dt.days.to_numpy()
        for name, c in terms.items():
            c = np.where(free & np.isfinite(c), c, 0.0)
            for w in WS: r[f"{name} w{w}"] = rm(p + np.clip(w * c, -CLIP, CLIP), y) - r0
            if name == "all25":
                q = p + np.clip(0.2 * c, -CLIP, CLIP); r["holiday rows w0.2"] = rm(q[hol], y[hol]) - rm(p[hol], y[hol]); r["ordinary rows w0.2"] = rm(q[~hol], y[~hol]) - rm(p[~hol], y[~hol])
                e = lead <= 61; r["lead<=61 w0.2"] = rm(q[e], y[e]) - rm(p[e], y[e]); r["lead>61 w0.2"] = (rm(q[~e], y[~e]) - rm(p[~e], y[~e])) if (~e).any() else np.nan
                r["corr(error, term)"] = float(np.corrcoef((y - p)[free], c[free])[0, 1])
        res[str(origin.date())] = r
    T = pd.DataFrame(res).T; show = T.drop(columns=["corr(error, term)"]) * 1e4; pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    print(show.round(2).to_string()); print("corr(error, term):", T["corr(error, term)"].round(3).to_dict())
    print("mean", show.mean().round(2).to_dict()); print("worse", (show > 0).sum().to_dict())
    json.dump(res, open("rail_shift.json", "w"), indent=1)
