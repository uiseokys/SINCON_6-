import sys, pickle
T = sys.argv[1]; out = sys.argv[2]
sys.path.insert(0, T); sys.dont_write_bytecode = True
import pandas as pd
import ensemble as en, improved as im
from make_calendar import build_calendar
meta = pd.read_csv(f"{T}/data/metaData.csv")[["station_id", "canonical_line"]].rename(columns={"canonical_line": "line"})
te = im.enrich(pd.read_csv(f"{T}/data/test.csv"), build_calendar(), meta, True)
assert len(te) == 157194 and str(te["date"].min())[:10] == "2026-07-01" and str(te["date"].max())[:10] == "2026-10-31"
miss = te[list(en.FEATURES)].isna().mean(); print("features with NaN share > 0:", miss[miss > 0].round(3).to_dict())
pickle.dump({"df": te, "features": list(en.FEATURES)}, open(out, "wb")); print(te.shape)
