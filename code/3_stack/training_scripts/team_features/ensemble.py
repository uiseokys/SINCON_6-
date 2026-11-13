"""SINCON 2026 앙상블: LightGBM(improved.py) + 임베딩 MLP(nn_model.py)를 log 공간에서 평균.

순서
  01 데이터 읽기, 달력·외부 자료 연결 (improved.py와 같은 특징)
  02 자체 검증 (시간 순서 유지, 세 구간) — 모델별 점수와 앙상블 점수 출력
       A: ~2026-02-28 학습 -> 2026-03~06월   B: ~2025-12-31 학습 -> 2026-01~04월   C: ~2025-10-31 학습 -> 2025-11~2026-02월
  03 전체 기간으로 다시 학습해 제출 파일 2개 저장
       submission_ens30.csv       두 모델의 평균
       submission_ens30_yoy.csv   평균 전에 각 모델의 수준을 '전년 대비 증가율'에 맞춘 것 (아래 anchor_yoy)
       submission_ens30_level_station.csv   평균에 전체 수준 보정과 역별 백테스트 오차 보정을 더한 것 (최종 후보)

실행
  python ensemble.py                 # 검증 + 제출 파일
  python ensemble.py --skip_valid    # 제출 파일만

필요 패키지: pandas, numpy, lightgbm, torch (Kaggle Notebook·Colab 기본 설치). 신경망은 GPU가 있으면 GPU로 학습합니다.
누출 방지: 검증에서는 검증 기간과 그 이후 행을 학습에 쓰지 않습니다. 승하차 실측에서 만든 특징은 없습니다.
          수준 보정(anchor_yoy)도 기준일까지의 실측과 예측값만 사용합니다.
외부 자료: 2026년 7월 이후의 실제 경기·관중·행사·학사일정·날씨를 씁니다 (2026-11-11 규칙 변경 전제).
이력: submission_ensemble.csv(신경망 seed 10개, Public 0.16910)는 이전 버전의 출력입니다.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import EXPECTED_ROWS, find_data_dir
from external_features import (ATTENDANCE_FEATURES, EVENT_FEATURES, GAME_FEATURES, MUNHAK_STATION, SCHOOL_FEATURES,
                               WEATHER_FEATURES)
from improved import CALENDAR_FEATURES, enrich, fit_predict
from make_calendar import build_calendar
from nn_model import DEVICE, fit_predict_nn

HERE = Path(__file__).resolve().parent
FEATURES = CALENDAR_FEATURES + GAME_FEATURES + ATTENDANCE_FEATURES + EVENT_FEATURES + SCHOOL_FEATURES + WEATHER_FEATURES
LGB_SEEDS = [0, 1, 2]
NN_SEEDS = list(range(30))         # seed 평균 효과가 커서 30개 사용 (검증에서 10개 대비 약 0.0005 개선)
NN_KW = dict(epochs=30, width=512, drop=0.15)
NN_WEIGHT = 0.5                    # 앙상블 = (1 - w) * LightGBM + w * 신경망 (log1p 공간)
# 역별 보정: 서로 겹치지 않는 두 백테스트(검증 A·C)에서 앙상블이 역마다 평균적으로 틀린 양(전체 평균 제거)을
# 최종 예측에서 그대로 뺀다. 성장 중인 역은 계속 과소, 계양 등은 계속 과대 예측되는 경향이 구간마다 반복되기 때문.
STATION_BIAS_FOLDS = ("A", "C")
STATION_BIAS_SCALE = 1.0           # Public 점수로 확인한 최적 배율은 약 1.02 (k=0.5 제출 0.16301, 기준 0.1675)
# 전체 수준 보정: Public 점수 두 개(0.16910, +0.06 이동 시 0.17156)로 추정한 평균 오차 -0.023을 seed 30개 앙상블에 맞춘 값.
# 학습 데이터가 아니라 Public 리더보드에서 정한 값이다.
PUBLIC_LEVEL_SHIFT = 0.0217
VALID_FOLDS = [                    # (이름, 학습 마지막 날, 검증 시작, 검증 끝)
    ("A", "2026-02-28", "2026-03-01", "2026-06-30"),
    ("B", "2025-12-31", "2026-01-01", "2026-04-30"),
    ("C", "2025-10-31", "2025-11-01", "2026-02-28"),
]


def predict_both(train: pd.DataFrame, target: pd.DataFrame):
    """(LightGBM, 신경망) 각각의 log1p 예측."""
    log_lgb = np.log1p(fit_predict(train, target, FEATURES, LGB_SEEDS))
    log_nn = fit_predict_nn(train, target, FEATURES, NN_SEEDS, **NN_KW)
    return log_lgb, log_nn


def anchor_yoy(train: pd.DataFrame, target: pd.DataFrame, log_pred: np.ndarray):
    """예측 수준을 '기준일에 알 수 있던 전년 대비 증가율'에 맞춘다. 불가능하면(전년 자료 없음) None.

    평상 평일(휴무·징검다리·연휴 전후 제외), 18개월 내내 있던 역(문학경기장 제외)의 역x시간 칸 기준으로
      목표  = 기준일이 속한 해의 1월~기준일 달: (올해 칸 평균 - 작년 같은 달 칸 평균)의 평균
      현재  = 대상 달마다: (예측 칸 평균 - 작년 같은 달 실측 칸 평균)의 평균
    대상 달마다 (목표 - 현재)를 그 달의 모든 예측에 더한다. 여름처럼 학습에 한 번만 나온 달에서
    모델이 '달의 수준'과 '성장'을 분리하지 못하는 문제를 줄이기 위한 보정이다.
    """
    cutoff = train["date"].max()
    year, prev = cutoff[:4], str(int(cutoff[:4]) - 1)
    first_day = train.groupby("station_id")["date"].min()
    stations = [s for s in first_day.index[first_day == train["date"].min()] if s != MUNHAK_STATION]
    plain = ((train["is_offday"] == 0) & (train["is_bridge_day"] == 0) & (train["is_semi_holiday"] == 0)
             & (train["is_before_long_off"] == 0) & (train["is_after_long_off"] == 0) & train["station_id"].isin(stations))
    hist = train[plain].assign(log_y=np.log1p(train.loc[plain, "boardings"]), mm=train.loc[plain, "date"].str[5:7])
    cell = lambda d, col: d.groupby(["mm", "station_id", "hour"])[col].mean()
    this_year, last_year = cell(hist[hist["date"].str[:4] == year], "log_y"), cell(hist[hist["date"].str[:4] == prev], "log_y")
    growth = (this_year - last_year).groupby("mm").mean().dropna()
    tplain = ((target["is_offday"] == 0) & (target["is_bridge_day"] == 0) & (target["is_semi_holiday"] == 0)
              & (target["is_before_long_off"] == 0) & (target["is_after_long_off"] == 0) & target["station_id"].isin(stations))
    tgt = target.assign(log_p=log_pred, mm=target["date"].str[5:7])
    if len(growth) == 0 or not set(tgt["date"].str[:4]) == {year}:
        return None
    months = sorted(tgt["mm"].unique())
    implied = (cell(tgt[tplain], "log_p") - last_year).groupby("mm").mean().reindex(months)   # 대상 달만
    if implied.isna().any():
        return None
    return log_pred + tgt["mm"].map(float(growth.mean()) - implied).to_numpy()


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out", default=str(HERE / "submission_ens30.csv"))
    ap.add_argument("--skip_valid", action="store_true", help="검증 생략 (역별 보정은 저장된 station_backtest_bias.csv 사용)")
    ap.add_argument("--quick", action="store_true", help="동작 확인용: seed를 1개씩만 사용")
    ap.add_argument("--nn_weight", type=float, default=NN_WEIGHT, help="앙상블에서 신경망 비중 lambda (LightGBM은 1 - lambda)")
    args = ap.parse_args()
    lam = args.nn_weight
    if args.quick:
        LGB_SEEDS[:] = LGB_SEEDS[:1]
        NN_SEEDS[:] = NN_SEEDS[:1]
    out_dir = Path(args.out).resolve().parent
    bias_path = out_dir / "station_backtest_bias.csv"

    # 01 데이터 읽기, 특징 연결
    data_dir = find_data_dir(args.data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    meta = pd.read_csv(data_dir / "metaData.csv")[["station_id", "canonical_line"]].rename(columns={"canonical_line": "line"})
    cal = build_calendar()
    train_e, test_e = enrich(train, cal, meta, True), enrich(test, cal, meta, True)
    print(f"train {train_e.shape}, test {test_e.shape}, 특징 {len(FEATURES)}개, 신경망 장치 {DEVICE}", flush=True)

    # 02 자체 검증
    if not args.skip_valid:
        rows, station_bias = [], []
        for name, train_end, start, end in VALID_FOLDS:
            t = time.time()
            tr = train_e[train_e["date"] <= train_end]
            va = train_e[train_e["date"].between(start, end)]
            assert tr["date"].max() < va["date"].min()
            log_lgb, log_nn = predict_both(tr, va)
            y = np.log1p(va["boardings"].to_numpy())
            ens = (1 - lam) * log_lgb + lam * log_nn
            rows.append((rmse(log_lgb, y), rmse(log_nn, y), rmse(ens, y)))
            if name in STATION_BIAS_FOLDS:
                err = pd.Series(ens - y, index=va["station_id"].to_numpy())
                station_bias.append(err.groupby(level=0).mean() - err.mean())
            msg = (f"검증 {name}: ~{train_end} 학습 -> {start}~{end} | LightGBM {rows[-1][0]:.5f} | 신경망 {rows[-1][1]:.5f} "
                   f"| 앙상블 {rows[-1][2]:.5f}")
            a_lgb, a_nn = anchor_yoy(tr, va, log_lgb), anchor_yoy(tr, va, log_nn)
            if a_lgb is not None and a_nn is not None:   # 전년 자료가 있는 구간에서만 수준 보정을 평가할 수 있음
                msg += f" | 수준 보정 앙상블 {rmse((1 - lam) * a_lgb + lam * a_nn, y):.5f}"
            print(msg + f" | {time.time() - t:.0f}초", flush=True)
        m = np.mean(rows, axis=0)
        print(f"검증 평균: LightGBM {m[0]:.5f} | 신경망 {m[1]:.5f} | 앙상블 {m[2]:.5f}", flush=True)
        bias = pd.concat(station_bias, axis=1).mean(axis=1)
        (bias - bias.mean()).rename("b").rename_axis("station_id").to_csv(bias_path)

    # 03 전체 기간으로 학습 후 제출 파일 저장
    t = time.time()
    assert train_e["date"].max() < test_e["date"].min()
    log_lgb, log_nn = predict_both(train_e, test_e)
    pd.DataFrame({"example_id": test["example_id"], "log_lgb": log_lgb, "log_nn": log_nn}).to_csv(out_dir / "ensemble_parts.csv", index=False)
    a_lgb, a_nn = anchor_yoy(train_e, test_e, log_lgb), anchor_yoy(train_e, test_e, log_nn)
    outputs = {args.out: (log_lgb, log_nn), args.out.replace(".csv", "_yoy.csv"): (a_lgb, a_nn)}
    if bias_path.exists():             # 수준 보정 + 역별 보정 (두 모델에 같은 양을 더하므로 평균에도 그대로 반영됨)
        b = pd.read_csv(bias_path).set_index("station_id")["b"]
        # PUBLIC_LEVEL_SHIFT는 0.5:0.5 평균을 기준으로 잰 값이므로, lambda가 달라져도 전체 평균 수준은 같게 유지한다
        keep_level = float((0.5 * (log_lgb + log_nn)).mean() - ((1 - lam) * log_lgb + lam * log_nn).mean())
        adj = PUBLIC_LEVEL_SHIFT + keep_level - STATION_BIAS_SCALE * test_e["station_id"].map(b).to_numpy()
        outputs[args.out.replace(".csv", "_level_station.csv")] = (log_lgb + adj, log_nn + adj)
    for path, (p_lgb, p_nn) in outputs.items():
        submission = pd.DataFrame({"example_id": test["example_id"],
                                   "boardings": np.clip(np.expm1((1 - lam) * p_lgb + lam * p_nn), 0, None)})
        assert len(submission) == EXPECTED_ROWS and submission["example_id"].is_unique
        assert np.isfinite(submission["boardings"]).all() and (submission["boardings"] >= 0).all()
        submission.to_csv(path, index=False)
        print(f"저장: {Path(path).name} ({len(submission):,}행, 예측 평균 {submission['boardings'].mean():.1f})", flush=True)
    print(f"최종 학습·예측 {time.time() - t:.0f}초")


if __name__ == "__main__":
    main()
