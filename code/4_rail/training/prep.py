"""팀 신경망 입력을 한 번 만들어 저장 (2026-06-30 이전 train 만 사용)."""
import sys, pickle
T = sys.argv[1]; out = sys.argv[2]
sys.path.insert(0, T)
import pandas as pd
import ensemble as en, improved as im
from make_calendar import build_calendar
meta = pd.read_csv(f"{T}/data/metaData.csv")[["station_id", "canonical_line"]].rename(columns={"canonical_line": "line"})
tr = im.enrich(pd.read_csv(f"{T}/data/train.csv"), build_calendar(), meta, True)
assert str(tr["date"].max())[:10] == "2026-06-30"
pickle.dump({"df": tr, "features": list(en.FEATURES), "nn_kw": dict(en.NN_KW)}, open(out, "wb"))
print(tr.shape, len(en.FEATURES), en.NN_KW)
