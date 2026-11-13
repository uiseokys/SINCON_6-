"""SINCON 2026 베이스라인 개선 (OT 자료 8쪽): 달력 특징을 붙여 LightGBM으로 예측.

순서
  01 데이터 읽기
  02 달력 정보(공휴일·방학)와 월·요일·주말·계절, 외부 자료(문학 야구 경기·관중, 대형 행사, 날씨)를 붙여
     train_enriched.csv, test_enriched.csv 저장
  03 자체 검증 (시간 순서 유지, 두 구간)
       A: 2026-02-28까지 학습 -> 2026-03~06월 (baseline.py의 검증 대상과 같음)
       B: 2025-12-31까지 학습 -> 2026-01~04월 (설 연휴·겨울방학 포함)
  04 전체 기간으로 다시 학습해 제출 파일 저장

실행
  python improved.py                      # 달력 + 경기·관중·행사·학교 방학·날씨 -> submission_external_v3.csv
  python improved.py --no_external        # 달력만 (규칙 변경 전 기준) -> submission_improved.csv
  python improved.py --skip_valid         # 제출 파일만
  python improved.py --no_save_enriched   # 큰 CSV 2개는 저장하지 않음

필요 패키지: pandas, numpy, lightgbm (Kaggle Notebook·Colab 기본 설치)
누출 방지: 검증에서는 검증 기간과 그 이후 행을 학습에 쓰지 않습니다. 특징은 날짜·역·달력·경기·관중·행사·날씨뿐이고
          승하차 실측에서 만든 값은 없습니다.
외부 자료: 경기·관중·행사·날씨는 2026년 7월 이후 실제 값을 포함합니다(2026-11-11 규칙 변경 전제, external_features.py 참고).
이력: submission_external.csv(경기·날씨, Public 0.17120), submission_external_v2.csv(+관중·행사, Public 0.16953)는
      이전 버전의 출력입니다. v3는 NEIS 방학 비율, 여름 성수기, 명절 상대 일수를 더했습니다.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from baseline import EXPECTED_ROWS, find_data_dir, rmsle
from external_features import (ATTENDANCE_FEATURES, EVENT_FEATURES, GAME_FEATURES, SCHOOL_FEATURES,
                               WEATHER_FEATURES, add_external)
from make_calendar import build_calendar

HERE = Path(__file__).resolve().parent

CALENDAR_FEATURES = [
    # 역·시간
    "station_id", "line", "hour",
    # 월·요일·주말·계절
    "dow", "month", "is_weekend", "season",
    # 공휴일·연휴
    "holiday_kind", "is_holiday", "is_lunar_day", "is_semi_holiday", "is_offday", "is_weekday_holiday",
    "offday_block_len", "offday_block_pos", "is_lunar_block", "is_bridge_day",
    "is_before_long_off", "is_after_long_off", "days_to_offday", "days_from_offday",
    # 방학·휴가철
    "is_school_vacation", "is_univ_vacation", "is_summer_peak",
    # 설날·추석 당일 기준 며칠 전후 (그 밖은 NaN)
    "lunar_rel_day",
    # 시간 축 (승객 증가, 신규역 정착, 2025-06-28 연장 개통에 따른 수준 변화)
    "days_since_start",
]
CATEGORICAL = ["station_id"]

PARAMS = dict(
    objective="regression",        # log1p(승차)에 대한 L2 = RMSLE와 같은 기준
    learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    deterministic=True, force_row_wise=True, verbose=-1,
)
NUM_ROUNDS = 1500                  # 고정값. 최종 학습에는 검증 세트가 없어 early stopping을 쓰지 않음
SEEDS = [0, 1, 2]                  # 예측은 seed 3개 평균
VALID_FOLDS = [                    # (이름, 학습 마지막 날, 검증 시작, 검증 끝)
    ("A", "2026-02-28", "2026-03-01", "2026-06-30"),
    ("B", "2025-12-31", "2026-01-01", "2026-04-30"),
]


def enrich(df: pd.DataFrame, cal: pd.DataFrame, meta: pd.DataFrame, external: bool) -> pd.DataFrame:
    """날짜·역 기준으로 달력, 노선, (선택) 경기·날씨를 붙인다. 행 수와 순서는 그대로."""
    df = df.copy()
    df["date"] = df["date"].astype(str)
    out = df.merge(cal, on="date", how="left", validate="many_to_one")
    out = out.merge(meta, on="station_id", how="left", validate="many_to_one")
    if external:
        out = add_external(out)
    assert len(out) == len(df) and out[[c for c in CALENDAR_FEATURES if c != "lunar_rel_day"]].notna().all().all()
    return out


def fit_predict(train: pd.DataFrame, target: pd.DataFrame, features: list, seeds) -> np.ndarray:
    """log1p(승차)를 학습하고 target 행의 예측을 승차 인원 단위로 돌려준다 (seed 평균)."""
    y = np.log1p(train["boardings"].to_numpy())
    log_pred = np.zeros(len(target))
    for seed in seeds:
        data = lgb.Dataset(train[features], y, categorical_feature=CATEGORICAL)
        model = lgb.train({**PARAMS, "seed": seed}, data, NUM_ROUNDS)
        log_pred += model.predict(target[features]) / len(seeds)
    return np.clip(np.expm1(log_pred), 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_external", action="store_true", help="경기·날씨를 쓰지 않음 (규칙 변경 전 기준)")
    ap.add_argument("--skip_valid", action="store_true")
    ap.add_argument("--no_save_enriched", action="store_true")
    args = ap.parse_args()
    external = not args.no_external
    external_features = GAME_FEATURES + ATTENDANCE_FEATURES + EVENT_FEATURES + SCHOOL_FEATURES + WEATHER_FEATURES
    features = CALENDAR_FEATURES + (external_features if external else [])
    out_path = args.out or str(HERE / ("submission_external_v3.csv" if external else "submission_improved.csv"))

    # 01 데이터 읽기
    data_dir = find_data_dir(args.data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    meta = pd.read_csv(data_dir / "metaData.csv")[["station_id", "canonical_line"]].rename(columns={"canonical_line": "line"})

    # 02 달력·외부 정보 연결
    cal = build_calendar()
    cal.to_csv(HERE / "calendar_features.csv", index=False)
    train_e, test_e = enrich(train, cal, meta, external), enrich(test, cal, meta, external)
    # 경기·관중·행사·명절 상대 일수 열은 해당 없는 날이 NaN이므로 나머지만 확인
    must_have = [c for c in CALENDAR_FEATURES if c != "lunar_rel_day"] + (WEATHER_FEATURES + SCHOOL_FEATURES if external else [])
    assert test_e[must_have].notna().all().all()
    if not args.no_save_enriched:
        train_e.to_csv(HERE / "train_enriched.csv", index=False)
        test_e.to_csv(HERE / "test_enriched.csv", index=False)
    print(f"train_enriched {train_e.shape}, test_enriched {test_e.shape}, 특징 {len(features)}개 (외부 자료 {'사용' if external else '미사용'})")

    # 03 자체 검증 (검증 기간과 그 이후는 학습에서 제외)
    if not args.skip_valid:
        scores = []
        for name, train_end, start, end in VALID_FOLDS:
            t = time.time()
            tr = train_e[train_e["date"] <= train_end]
            va = train_e[train_e["date"].between(start, end)]
            assert tr["date"].max() < va["date"].min()
            pred = fit_predict(tr, va, features, SEEDS)
            y = va["boardings"].to_numpy()
            hol = (va["is_holiday"] == 1).to_numpy()
            scores.append(rmsle(y, pred))
            print(f"검증 {name}: ~{train_end} 학습 -> {start}~{end} RMSLE {scores[-1]:.5f} "
                  f"| 공휴일 {rmsle(y[hol], pred[hol]):.4f}, 그 외 {rmsle(y[~hol], pred[~hol]):.4f} | {time.time() - t:.0f}초")
        print(f"검증 평균 RMSLE {np.mean(scores):.5f}  (baseline.py는 검증 A에서 0.30770)")

    # 04 전체 기간으로 학습 후 제출 파일 저장
    t = time.time()
    assert train_e["date"].max() < test_e["date"].min()
    test_e["boardings"] = fit_predict(train_e, test_e, features, SEEDS)
    submission = test_e[["example_id", "boardings"]]
    assert len(submission) == EXPECTED_ROWS and submission["example_id"].is_unique
    assert (submission["example_id"].to_numpy() == test["example_id"].to_numpy()).all()
    assert np.isfinite(submission["boardings"]).all() and (submission["boardings"] >= 0).all()
    submission.to_csv(out_path, index=False)
    print(f"저장: {out_path} ({len(submission):,}행, 예측 평균 {submission['boardings'].mean():.1f}) | {time.time() - t:.0f}초")


if __name__ == "__main__":
    main()
