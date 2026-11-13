"""Step 2 notebook support. Standard library, numpy and pandas only; no network."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform

import numpy as np
import pandas as pd

ROWS = 157_194
BASELINE_TOLERANCE = 1e-9
REPRODUCTION_TOLERANCE = 1e-10


def bounded_matches(root, filename, max_depth, skip_names=()):
    """Search at most max_depth directories down; never follow symlinks."""
    root = Path(root)
    if not root.is_dir():
        return []
    found = []
    for directory, children, files in os.walk(root, followlinks=False):
        current = Path(directory)
        depth = len(current.relative_to(root).parts)
        children[:] = sorted(name for name in children
                             if name not in skip_names and depth < max_depth
                             and not (current / name).is_symlink())
        if filename in files:
            found.append(current / filename)
    return sorted(found)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path, **kwargs):
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_inputs(package):
    manifest = json.loads((package / "delivery_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("잘못된 매니페스트 경로")
        if sha256(package / relative) != expected:
            raise ValueError("동결 입력 해시 불일치: " + name)
    return manifest


def find_data(package):
    """Prefer the designated folder. Permit separate official Kaggle Datasets."""
    expected = json.loads((package / "official_inputs.json").read_text(encoding="utf-8"))
    direct = [package / "data", package, Path.cwd() / "data"]
    candidates = list(direct)
    for root, depth, skipped in ((Path("/kaggle/input"), 4, ()),
                                 (Path("/content"), 3, ("drive",))):
        candidates.extend(p.parent for p in bounded_matches(root, "train.csv", depth, skipped))
    for directory in dict.fromkeys(candidates):
        if all((directory / name).is_file() for name in expected):
            if all(sha256(directory / name) == item["sha256"] for name, item in expected.items()):
                return directory.resolve(), expected
    raise FileNotFoundError(
        "공식 train.csv, test.csv, metaData.csv를 패키지의 data/에 넣으세요. "
        "Kaggle에서는 공식 대회 Dataset을 별도로 연결할 수도 있습니다. "
        "파일명은 대소문자까지 같아야 하며 official_inputs.json의 해시와 일치해야 합니다. "
        "자동 탐색은 /content 깊이 3, drive 제외, /kaggle/input 깊이 4로 제한됩니다. "
        "못 찾으면 ROOT에 패키지 폴더를 지정하고 그 data/에 공식 CSV를 놓으세요."
    )


def validate_submission(frame, grid):
    if list(frame) != ["example_id", "boardings"]:
        raise ValueError("제출 열은 example_id,boardings 순서여야 합니다")
    if len(frame) != ROWS or not frame.example_id.is_unique:
        raise ValueError("157,194개의 고유 ID가 필요합니다")
    if not frame.example_id.equals(grid.example_id):
        raise ValueError("공식 test.csv와 example_id 순서가 다릅니다")
    values = frame.boardings.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("예측은 유한한 비음수여야 합니다")
    return dict(rows=len(frame), ids_in_official_order=True, unique_ids=True,
                finite=True, nonnegative=True)


def compare(frame, reference_path, grid):
    reference = read_csv(reference_path)
    validate_submission(reference, grid)
    difference = np.abs(np.log1p(frame.boardings.to_numpy()) - np.log1p(reference.boardings.to_numpy()))
    return dict(reference=Path(reference_path).name, reference_sha256=sha256(reference_path),
                max_abs_log1p_difference=float(difference.max()),
                rows_boardings_different=int((frame.boardings.to_numpy() != reference.boardings.to_numpy()).sum()),
                rows_nonzero_log1p_difference=int((difference > 0).sum()),
                rows_log_difference_gt_1e_10=int((difference > 1e-10).sum()))


def verify_baseline(baseline, reference_path, grid, expected_sha256):
    """Accept identical bytes or a valid CSV within the fixed numerical tolerance."""
    frame = read_csv(baseline)
    validation = validate_submission(frame, grid)
    comparison = compare(frame, reference_path, grid)
    actual_sha256 = sha256(baseline)
    exact = actual_sha256 == expected_sha256
    difference = comparison["max_abs_log1p_difference"]
    if not exact and difference > BASELINE_TOLERANCE:
        raise ValueError(
            f"1단계 CSV의 최대 log1p 차이 {difference:.17g}가 허용 {BASELINE_TOLERANCE:g}를 넘습니다. "
            "full 재학습 결과는 별도 검토가 필요합니다."
        )
    record = dict(accepted=True, acceptance="sha256_exact" if exact else "numeric_equivalence",
                  input_sha256=actual_sha256, reference_sha256=expected_sha256,
                  max_abs_log1p_difference=difference,
                  rows_boardings_different=comparison["rows_boardings_different"],
                  rows_nonzero_log1p_difference=comparison["rows_nonzero_log1p_difference"],
                  tolerance=BASELINE_TOLERANCE, calculation_uses_supplied_csv=True, **validation)
    if not exact:
        print("1단계 입력: SHA는 다르지만 수치 동등성 검사로 허용했습니다. 계산은 제공된 CSV로 진행합니다.")
        print(json.dumps(record, ensure_ascii=False, indent=2))
    return record


def save_submission(frame, target, grid, reference):
    validation = validate_submission(frame, grid)
    frame.to_csv(target, index=False, mode="x", lineterminator="\n")
    saved = read_csv(target)
    validate_submission(saved, grid)
    if not np.array_equal(saved.boardings.to_numpy(), frame.boardings.to_numpy()):
        raise ValueError("CSV 재읽기에서 값이 변했습니다")
    return dict(file=Path(target).name, sha256=sha256(target), **validation,
                comparison=compare(saved, reference, grid))


def prepare(package, baseline_override=""):
    package = Path(package).resolve()
    manifest = verify_inputs(package)
    data, official = find_data(package)
    meta = read_csv(data / "metaData.csv")
    grid = read_csv(data / "test.csv", parse_dates=["date"])
    if not meta.station_id.is_unique or not grid.station_id.isin(meta.station_id).all():
        raise ValueError("metaData.csv의 역 ID와 예측 격자가 맞지 않습니다")
    baseline = Path(baseline_override).expanduser() if baseline_override else (
        package / "input" / "submission_clean_refined.csv"
        if (package / "input" / "submission_clean_refined.csv").is_file()
        else package / "reference" / "submission_clean_refined.csv"
    )
    if not baseline.is_absolute():
        baseline = package / baseline
    baseline_verification = verify_baseline(
        baseline, package / "reference" / "submission_clean_refined.csv", grid, manifest["clean_sha256"])
    layers = load_module(package / "final_layers.py", "sincon_accepted_layers")
    output_root = (Path("/kaggle/working") if Path("/kaggle/working").is_dir()
                   else Path("/content") if Path("/content").is_dir()
                   else package / "outputs")
    output = output_root / ("sincon_step2_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    return dict(package=package, data=data, grid=grid, baseline=baseline, output=output,
                layers_module=layers, official=official, baseline_verification=baseline_verification,
                environment=dict(python=platform.python_version(), numpy=np.__version__,
                                 pandas=pd.__version__, system=platform.system(),
                                 machine=platform.machine(), processor=platform.processor(),
                                 random_seed="none; deterministic layer arithmetic", gpu_used=False))


def layer_table(deltas, report):
    return pd.DataFrame([
        dict(layer=name, enabled=name in report["enabled"],
             changed_rows=report["layers"].get(name, {}).get("changed_rows", 0),
             max_abs_log_delta=float(deltas[name].abs().max()))
        for name in ("timecapsule", "plave_existing", "station025", "friday", "plave22", "lunar3122")
    ])


def show_layer(name, deltas, report, limit=12):
    summary = layer_table(deltas, report)
    print(summary.loc[summary.layer.eq(name)].to_string(index=False))
    changed = deltas.loc[deltas[name].abs().gt(1e-12), ["example_id", "date", "station_id", "hour", name]]
    if len(changed):
        print(changed.head(limit).to_string(index=False))
        if len(changed) > limit:
            print(f"앞쪽 {limit}행만 표시합니다. 전체 행은 layer_deltas.csv.gz에 저장합니다.")


def apply_hedges(main, grid, package):
    """Reproduce the CURRENT optional candidate, never tune its assumptions."""
    source = load_module(package / "event_hedges.py", "sincon_unvalidated_hedges")
    validate_submission(main, grid)
    values = main.boardings.to_numpy(copy=True)
    rows = []
    judgments = read_csv(package / "evidence" / "hedges_15_current_review.csv")
    for event, date, audience, shares, weight in source.EVENTS:
        weekday = pd.Timestamp(date).dayofweek
        for hour, share in shares.items():
            matched = np.flatnonzero(grid.station_id.eq(source.STATION) & grid.date.eq(date) & grid.hour.eq(hour))
            if len(matched) != 1:
                raise ValueError("절충 대상 키가 유일하지 않습니다")
            i = matched[0]
            full = share if audience is None else source.BASE[weekday][hour] + source.RATIO * audience * share
            before = float(values[i])
            values[i] = float(np.exp((1 - weight) * np.log1p(values[i]) + weight * np.log1p(full)) - 1)
            original = judgments.loc[judgments.example_id.eq(grid.iloc[i].example_id)].iloc[0]
            rows.append(dict(example_id=grid.iloc[i].example_id, event=event, date=date, hour=hour,
                             current=before, hedged=float(values[i]), full=float(full),
                             audience=audience, ratio=source.RATIO if audience is not None else None,
                             share=share if audience is not None else None, weight=weight,
                             round11_judgment=original.judgment,
                             caveat="11번 지지는 방향·과거 규모에 한정; W의 독립 검증 아님"))
    result = main.assign(boardings=values)
    changed = values != main.boardings.to_numpy()
    if int(changed.sum()) != 15 or set(grid.loc[changed, "example_id"]) != set(judgments.example_id):
        raise ValueError("별도 후보는 지정된 15행만 바뀌어야 합니다")
    validate_submission(result, grid)
    return result, pd.DataFrame(rows)


def finish(context, main, deltas, report, enable_hedges=False):
    package, output, grid = context["package"], context["output"], context["grid"]
    records = {}
    main_name = "submission_night_v4_main.csv" if set(report["enabled"]) == set(context["layers_module"].LAYER_NAMES) else "EXPERIMENT_layers_disabled.csv"
    records["main"] = save_submission(main, output / main_name, grid,
                                      package / "reference" / "submission_night_v4_main.csv")
    if main_name == "submission_night_v4_main.csv" and records["main"]["comparison"]["max_abs_log1p_difference"] > REPRODUCTION_TOLERANCE:
        raise ValueError("주후보 재현 차이가 1e-10을 넘습니다. 별도 검토가 필요합니다.")
    records["from_clean"] = compare(main, package / "reference" / "submission_clean_refined.csv", grid)
    deltas.to_csv(output / "layer_deltas.csv.gz", index=False,
                  compression={"method": "gzip", "mtime": 0}, mode="x")
    if enable_hedges:
        if set(report["enabled"]) != set(context["layers_module"].LAYER_NAMES):
            raise ValueError("별도 후보 재현은 6개 층을 모두 켠 상태에서만 지원합니다")
        hedged, assumptions = apply_hedges(main, grid, package)
        records["hedged"] = save_submission(hedged, output / "submission_night_v5_events_hedged15.csv", grid,
                                             package / "reference" / "submission_night_v5_events_hedged15.csv")
        if records["hedged"]["comparison"]["max_abs_log1p_difference"] > REPRODUCTION_TOLERANCE:
            raise ValueError("15행 후보 재현 차이가 1e-10을 넘습니다")
        assumptions.to_csv(output / "hedges_15_current_assumptions.csv", index=False, mode="x")
        records["hedged"]["changed_rows_from_main"] = int((hedged.boardings != main.boardings).sum())
        records["hedged"]["changed_example_ids"] = grid.loc[hedged.boardings.ne(main.boardings), "example_id"].tolist()
    else:
        print("ENABLE_HEDGES=False: 검증되지 않은 절충 후보는 생성하지 않았습니다.")
    result = dict(environment=context["environment"], official_inputs=context["official"],
                  baseline_sha256=sha256(context["baseline"]),
                  baseline_verification=context["baseline_verification"], layers=report,
                  outputs=records, output_directory=output.name,
                  note="기준 파일과의 일치 검증이며 미래 정답에 대한 정확도 검증이 아닙니다.")
    (output / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))
    print("결과 폴더:", output)
    return result
