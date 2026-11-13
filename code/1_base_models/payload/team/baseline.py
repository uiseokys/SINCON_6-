"""SINCON 2026 베이스라인 (OT 자료 6~7쪽 방식 그대로).

방식
  1) 최근 약 8주에서 같은 역·요일·시간대 기록을 고른다.
  2) 승차 인원에 log1p를 적용해 평균을 낸다.
  3) expm1으로 승차 인원 단위로 되돌려 미래의 같은 역·요일·시간대에 적용한다.

실행
  python baseline.py            # 자체 검증 점수 출력 + submission.csv 저장
  python baseline.py --data_dir <CSV 폴더> --out submission.csv

입력: train.csv, test.csv (같은 폴더의 data/, 또는 Kaggle의 /kaggle/input 아래)
출력: submission.csv (example_id, boardings) 157,194행
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["station_id", "dow", "hour"]

# OT 자료에 적힌 구간 그대로
VALID_PATTERN = ("2026-01-05", "2026-02-28")   # 자체 검증: 이 구간의 패턴으로
VALID_TARGET = ("2026-03-01", "2026-06-30")    #            3~6월을 예측 (OT 기준 RMSLE 0.30770)
FINAL_PATTERN = ("2026-05-04", "2026-06-30")   # 최종 예측: 최근 약 8주
EXPECTED_ROWS = 157_194                        # 123일 x 71역 x 18시간


def find_data_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    kaggle = sorted(Path("/kaggle/input").glob("**/train.csv")) if Path("/kaggle/input").exists() else []
    if kaggle:
        return kaggle[0].parent
    here = Path(__file__).resolve().parent
    return here / "data" if (here / "data" / "train.csv").exists() else here


def add_dow(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["date"].astype(str)
    df["dow"] = pd.to_datetime(df["date"]).dt.day_of_week   # 0=월 ... 6=일
    return df


def fit_pattern(train: pd.DataFrame, start: str, end: str) -> dict:
    """[start, end] 구간의 역·요일·시간대별 log1p 평균과, 빈칸을 메울 대체값."""
    hist = train[train["date"].between(start, end)]
    log_y = np.log1p(hist["boardings"])
    return {
        "sdh": log_y.groupby([hist[c] for c in KEY]).mean().rename("pattern").reset_index(),
        # 대체값: 그 역·요일·시간대 기록이 구간 안에 없을 때만 쓰임 (역·시간 평균 -> 전체 평균)
        "sh": log_y.groupby([hist["station_id"], hist["hour"]]).mean().rename("fallback").reset_index(),
        "all": float(log_y.mean()),
    }


def predict(pattern: dict, rows: pd.DataFrame) -> np.ndarray:
    """rows(date, station_id, hour, dow)에 패턴을 붙여 승차 인원 단위 예측을 돌려준다."""
    out = rows[["station_id", "dow", "hour"]].merge(pattern["sdh"], on=KEY, how="left")
    out = out.merge(pattern["sh"], on=["station_id", "hour"], how="left")
    log_pred = out["pattern"].fillna(out["fallback"]).fillna(pattern["all"]).to_numpy()
    assert len(log_pred) == len(rows)
    return np.clip(np.expm1(log_pred), 0, None)


def rmsle(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "submission.csv"))
    args = ap.parse_args()

    # 01 데이터 읽기
    data_dir = find_data_dir(args.data_dir)
    train = add_dow(pd.read_csv(data_dir / "train.csv"))
    test = add_dow(pd.read_csv(data_dir / "test.csv"))
    print(f"data_dir = {data_dir}")
    print(f"train {train.shape} {train['date'].min()} ~ {train['date'].max()} | test {test.shape} {test['date'].min()} ~ {test['date'].max()}")

    # 02 과거 구간으로 자체 검증 (검증 대상 기간의 정답은 패턴 계산에 쓰지 않음)
    assert VALID_PATTERN[1] < VALID_TARGET[0]
    valid = train[train["date"].between(*VALID_TARGET)]
    valid_pred = predict(fit_pattern(train, *VALID_PATTERN), valid)
    print(f"자체 검증: {VALID_PATTERN[0]}~{VALID_PATTERN[1]} 패턴 -> {VALID_TARGET[0]}~{VALID_TARGET[1]} "
          f"RMSLE {rmsle(valid['boardings'].to_numpy(), valid_pred):.5f}  (OT 자료 0.30770)")

    # 03 최종 예측 후 제출 파일 저장 (ID는 잘린 sample_submission이 아니라 test.csv 기준)
    assert FINAL_PATTERN[1] < test["date"].min()
    test["boardings"] = predict(fit_pattern(train, *FINAL_PATTERN), test)
    submission = test[["example_id", "boardings"]]
    assert len(submission) == EXPECTED_ROWS and submission["example_id"].is_unique
    assert np.isfinite(submission["boardings"]).all() and (submission["boardings"] >= 0).all()
    submission.to_csv(args.out, index=False)
    print(f"저장: {args.out}  ({len(submission):,}행, 예측 평균 {submission['boardings'].mean():.1f}, "
          f"최소 {submission['boardings'].min():.2f}, 최대 {submission['boardings'].max():.1f})")


if __name__ == "__main__":
    main()
