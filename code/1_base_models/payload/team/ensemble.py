from __future__ import annotations
from nn_model import DEVICE, fit_predict_nn
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from baseline import EXPECTED_ROWS, find_data_dir
from external_features import ATTENDANCE_FEATURES, EVENT_FEATURES, GAME_FEATURES, MUNHAK_STATION, SCHOOL_FEATURES, WEATHER_FEATURES
from improved import CALENDAR_FEATURES, enrich, fit_predict
from make_calendar import build_calendar
HERE = Path(__file__).resolve().parent
FEATURES = CALENDAR_FEATURES + GAME_FEATURES + ATTENDANCE_FEATURES + EVENT_FEATURES + SCHOOL_FEATURES + WEATHER_FEATURES
LGB_SEEDS = [0, 1, 2]
NN_SEEDS = list(range(30))
NN_KW = dict(epochs=30, width=512, drop=0.15)

def predict_both(train: pd.DataFrame, target: pd.DataFrame):
    """(LightGBM, 신경망) 각각의 log1p 예측."""
    log_lgb = np.log1p(fit_predict(train, target, FEATURES, LGB_SEEDS))
    log_nn = fit_predict_nn(train, target, FEATURES, NN_SEEDS, **NN_KW)
    return (log_lgb, log_nn)

def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))
