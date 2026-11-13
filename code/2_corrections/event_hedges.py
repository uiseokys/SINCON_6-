"""문학경기장역 대형 행사 3건의 '검증되지 않은' 절충 보정 (별도 후보용).

주후보(submission_night_v4_main.csv)에는 넣지 않는다. 과거 사례로 시간 순서 검증을 통과시키지 못한 추정이기 때문이다.
대신 "주후보 + 이 보정" 파일을 따로 만들어, 최종 2개 선택 중 하나로 쓸지 사용자가 정할 수 있게 한다.
두 파일은 지정된 15행만 다르다. PLAVE 22시 두 행은 주후보를 유지한다.

근거 (모두 2026-06-30 이전 승차 자료와 공개된 행사 사실. Public 점수와 7월 이후 지하철 실측은 쓰지 않음)
  - 이 역의 2026-03-28~06-30 관중 기록이 있는 야구 전 36경기. 이 중 매진은 11경기다.
    하루 초과 승차/관중의 중앙값은 약 0.31이다. 아래 기존 0.30을 새로 적합하지 않았다.
  - 과거 주경기장 공연 실제는 9일. 과거 예측 비교는 고유 7일만 가능하며 2025-03-20·21은 OOF가 없다.
  - 2025-10-24 랜더스 팬 행사: 21시 5,279명, 22시 1,140명 (평소 대비 +6,900명)

행사별 추정값 FULL과 혼합 W: 새 값 = (1+현재)^(1-W) × (1+FULL)^W - 1
  1. PLAVE 9/12·9/13 (예매 23,120 / 19,494건, 19:00 시작)
       21시 10% / 23시 5%만 계산한다. 22시 85%는 17행 이력의 가정이며 적용하지 않는다.
       (박효신 공연 3일의 시작 시각과 실제 비중 평균 약 14% / 84% / 2%만 참고. 공연 길이는 미확인.
        팀원이 조사한 버스 정류장 자료(국가교통DB)는 이용 조건이 확인되지 않아 이 계산에 쓰지 않았다.)
       W = 0.7
  2. Sustainable Wave Festival 8/29·8/30 (14:00 시작, 약 480분. 관객 수는 주최 측 전망 '이틀 4만 명'뿐이라 하루 1.2만 명으로 낮춰 가정)
       귀가 합계 = 0.30 x 12,000. 시간 비중 20시 10% / 21시 20% / 22시 45% / 23시 25% (종일 축제라 퇴장이 퍼진다고 가정)
       W = 0.5
  3. 랜더스페셜 매치 10/17 (23,000석 매진, 17:30 시작, 20:23경 종료)
       같은 시리즈 전년 행사(2025-10-24, 약 22,000명)의 실제 초과 승차를 관객 수 비례로 옮기고 종료 시각 차이(-20분)만큼 시간대를 보간한 값
       (Codex 7번 작업의 landers_same_series_diagnostic.csv): 20시 2,199 / 21시 4,102 / 22시 860명
       W = 0.5

실행: python event_hedges.py IN.csv OUT.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STATION = 3127
RATIO = 0.30
BASE = {5: {20: 150, 21: 63, 22: 21, 23: 5}, 6: {20: 37, 21: 24, 22: 15, 23: 4}}      # 평소 중앙값을 반올림한 기존 정수 표. 새 적합 없음.
EVENTS = [
    ("PLAVE", "2026-09-12", 23120, {21: 0.10, 23: 0.05}, 0.7),
    ("PLAVE", "2026-09-13", 19494, {21: 0.10, 23: 0.05}, 0.7),
    ("Sustainable Wave Festival", "2026-08-29", 12000, {20: 0.10, 21: 0.20, 22: 0.45, 23: 0.25}, 0.5),
    ("Sustainable Wave Festival", "2026-08-30", 12000, {20: 0.10, 21: 0.20, 22: 0.45, 23: 0.25}, 0.5),
    ("랜더스페셜 매치", "2026-10-17", None, {20: 2199.0, 21: 4102.0, 22: 860.0}, 0.5),      # attendance None: 값이 시간별 추정 승차
]


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    assert not out.exists(), "기존 파일을 덮어쓰지 않는다"
    test = pd.read_csv(HERE / "data" / "test.csv")
    sub = pd.read_csv(src)
    assert (sub["example_id"].to_numpy() == test["example_id"].to_numpy()).all() and len(sub) == 157194
    val = sub["boardings"].to_numpy().copy()
    for name, date, attendance, shares, w in EVENTS:
        dow = pd.Timestamp(date).dayofweek
        line = []
        for hour, share in shares.items():
            i = np.flatnonzero((test["station_id"] == STATION) & (test["date"] == date) & (test["hour"] == hour))[0]
            full = share if attendance is None else BASE[dow][hour] + RATIO * attendance * share
            new = float(np.exp((1 - w) * np.log1p(val[i]) + w * np.log1p(full)) - 1)
            line.append(f"{hour}시 {val[i]:.0f}->{new:.0f}(추정 {full:.0f})")
            val[i] = new
        print(f"{name} {date} (W={w}): " + ", ".join(line))
    changed = int((val != sub["boardings"].to_numpy()).sum())
    assert changed == 15, "별도 후보는 지정된 15행만 바뀌어야 한다"
    sub["boardings"] = val
    assert np.isfinite(val).all() and (val >= 0).all()
    sub.to_csv(out, index=False)
    print(f"저장 {out.name}: {changed}행 변경")


if __name__ == "__main__":
    main()
