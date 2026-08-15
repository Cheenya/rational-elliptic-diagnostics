from __future__ import annotations

import argparse
import csv
import hashlib
from io import StringIO
import json
from math import isclose, isfinite
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from elliptic_diagnostics.classifier import classify_a1_a6
from elliptic_diagnostics.curve import RationalCurve
from elliptic_diagnostics.experiment import CASE_ORDER, allocate_reference_quota, load_config


STAGE_A_COLUMNS = (
    "sample_id",
    "a",
    "b",
    "discriminant",
    "c2",
    "has3",
    "case",
    "three_torsion_x",
)
STAGE_A_METADATA_KEYS = {
    "backend",
    "case_counts",
    "include_zero_coefficients",
    "input_pairs",
    "nonsingular_rows",
    "seed",
    "singular_excluded",
}
STAGE_A_SUMMARY_COLUMNS = (
    "case",
    "c2",
    "has3",
    "count",
    "share_nonsingular",
    "input_pairs",
    "nonsingular_rows",
    "singular_excluded",
    "seed",
)
STAGE_B_COLUMNS = (
    "sample_id",
    "a",
    "b",
    "case",
    "torsion_order",
    "torsion_invariants",
    "generators",
)
STAGE_B_SELECTION_KEYS = {
    "available_per_case",
    "limit_per_case",
    "reference_total",
    "seed",
    "selected_per_case",
    "selected_total",
}
CALIBRATION_COLUMNS = (
    "backend",
    "n",
    "truth_positive",
    "truth_negative",
    "tp",
    "fp",
    "tn",
    "fn",
    "precision",
    "recall",
    "specificity",
    "accuracy",
    "input_sha256",
    "seed",
    "a_min",
    "a_max",
    "b_min",
    "b_max",
    "python_cpp_mismatches",
)
FINAL_RESULT_FILES = {
    "benchmark_scaling.csv",
    "calibration.csv",
    "environment.json",
    "stage_a_summary.csv",
    "stage_b_reference.csv",
}
BENCHMARK_COLUMNS = (
    "range_limit",
    "n",
    "repeats",
    "seed",
    "input_sha256",
    "python_wall_median_s",
    "python_cpu_median_s",
    "cpp_wall_median_s",
    "cpp_cpu_median_s",
    "sage_wall_median_s",
    "sage_cpu_median_s",
    "python_over_cpp_wall_x",
    "python_over_cpp_cpu_x",
    "sage_over_cpp_wall_x",
    "sage_over_cpp_cpu_x",
    "sage_over_python_wall_x",
    "sage_over_python_cpu_x",
    "two_torsion_candidates_mean",
    "three_torsion_candidates_mean",
    "exact_polynomial_checks_mean",
    "square_checks_mean",
    "python_cpp_mismatches",
    "python_sage_mismatches",
)
CANONICAL_BENCHMARK_ROWS = (
    (
        10000,
        8766499172605664842,
        "683a934bd6f44545cf2c7018c5dc6bd3b1de4884884c26c4dc6889295f516b1d",
    ),
    (
        30000,
        12352718890479622149,
        "0941364b6c3e1a624407e573683e80af6c4ec205202df60fe6e77652a6b8ba30",
    ),
    (
        100000,
        14501925109644847220,
        "e6ab143a45d3f40fb828d7d628a141f1e0cc7e9daee2b09906e458658d869eee",
    ),
    (
        300000,
        17547179935380367324,
        "38277e4182b052667dd24021db619fb75a536c42fb55f759e2912f495292474d",
    ),
)
ENVIRONMENT_KEYS = {
    "os",
    "architecture",
    "cpu_model",
    "logical_cores",
    "memory_gib",
    "sage_version",
    "python_version",
    "compiler_version",
}
CASE_PARAMETERS = {
    "A1": (0, 1),
    "A2": (1, 1),
    "A3": (3, 1),
    "A4": (0, 0),
    "A5": (1, 0),
    "A6": (3, 0),
}
SAMPLE_ID_PATTERN = re.compile(r"sample_([0-9]{9})\Z")


def _root_relative(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _read_csv_exact(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(columns):
            raise ValueError(f"invalid schema for {path.name}")
        rows = list(reader)
    for row in rows:
        if set(row) != set(columns) or any(value is None for value in row.values()):
            raise ValueError(f"invalid row schema for {path.name}")
    return rows


def _read_json_object(path: Path, keys: set[str]) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"invalid schema for {path.name}")
    return value


def _require_plain_int(value: Any, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _parse_csv_int(value: Any, field: str) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"-?(0|[1-9][0-9]*)", value):
        raise ValueError(f"invalid integer field {field}")
    return int(value)


def _parse_csv_float(value: Any, field: str, minimum: float) -> float:
    if not isinstance(value, str):
        raise ValueError(f"invalid numeric field {field}")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"invalid numeric field {field}") from error
    if not isfinite(parsed) or parsed < minimum:
        raise ValueError(f"invalid numeric field {field}")
    return parsed


def _validate_case_mapping(case: str, c2: int, has3: int) -> None:
    if case not in CASE_PARAMETERS or CASE_PARAMETERS[case] != (c2, has3):
        raise ValueError("inconsistent Stage A case/c2/has3 values")


def _validate_stage_a(
    input_dir: Path,
    config: Any,
) -> tuple[list[dict[str, str]], dict[str, Any], str]:
    rows = _read_csv_exact(input_dir / "stage_a_rows.csv", STAGE_A_COLUMNS)
    metadata = _read_json_object(
        input_dir / "stage_a_metadata.json",
        STAGE_A_METADATA_KEYS,
    )
    input_pairs = _require_plain_int(metadata["input_pairs"], "input_pairs", 0)
    nonsingular_rows = _require_plain_int(
        metadata["nonsingular_rows"],
        "nonsingular_rows",
        0,
    )
    singular_excluded = _require_plain_int(
        metadata["singular_excluded"],
        "singular_excluded",
        0,
    )
    if nonsingular_rows + singular_excluded != input_pairs:
        raise ValueError("Stage A metadata accounting mismatch")
    if nonsingular_rows != len(rows):
        raise ValueError("Stage A row count does not match metadata")
    if metadata["backend"] != "cpp17-cmake":
        raise ValueError("invalid Stage A backend")
    metadata_seed = _require_plain_int(
        metadata["seed"],
        "stage_a_metadata.seed",
    )
    if metadata_seed != config.seed:
        raise ValueError("Stage A seed does not match configuration")
    if metadata["include_zero_coefficients"] is not config.include_zero_coefficients:
        raise ValueError("Stage A zero-coefficient policy does not match configuration")
    if not isinstance(metadata["case_counts"], dict):
        raise ValueError("Stage A case_counts must be an object")
    if list(metadata["case_counts"]) != list(CASE_ORDER):
        raise ValueError("Stage A case_counts must use A1-A6 order")

    actual_counts = {case: 0 for case in CASE_ORDER}
    seen_ids: set[str] = set()
    previous_number = 0
    for row in rows:
        match = SAMPLE_ID_PATTERN.fullmatch(row["sample_id"])
        if match is None:
            raise ValueError("invalid Stage A sample_id")
        sample_number = int(match.group(1))
        if (
            row["sample_id"] in seen_ids
            or sample_number <= previous_number
            or sample_number > input_pairs
        ):
            raise ValueError("Stage A sample IDs are duplicated or unstable")
        seen_ids.add(row["sample_id"])
        previous_number = sample_number

        a = _parse_csv_int(row["a"], "a")
        b = _parse_csv_int(row["b"], "b")
        discriminant = _parse_csv_int(row["discriminant"], "discriminant")
        c2 = _parse_csv_int(row["c2"], "c2")
        has3 = _parse_csv_int(row["has3"], "has3")
        if not config.a_min <= a <= config.a_max or not config.b_min <= b <= config.b_max:
            raise ValueError("Stage A coefficient outside configured bounds")
        curve = RationalCurve(a=a, b=b)
        if not curve.is_nonsingular() or discriminant == 0:
            raise ValueError(f"singular Stage A row: {row['sample_id']}")
        if discriminant != curve.discriminant:
            raise ValueError("Stage A discriminant mismatch")
        if not config.include_zero_coefficients and (a == 0 or b == 0):
            raise ValueError("Stage A zero coefficient violates configuration")
        classification = classify_a1_a6(curve)
        expected_roots = sorted(
            candidate["x"]
            for candidate in classification.diagnostics["three_torsion_candidates"]
        )
        expected_roots_text = ",".join(str(root) for root in expected_roots) or "-"
        if (
            c2 != classification.c2
            or has3 != int(classification.has_3_torsion_indicator)
            or row["case"] != classification.case
            or row["three_torsion_x"] != expected_roots_text
        ):
            raise ValueError(f"Stage A classifier mismatch: {row['sample_id']}")
        _validate_case_mapping(row["case"], c2, has3)
        actual_counts[row["case"]] += 1

    for case in CASE_ORDER:
        expected = _require_plain_int(metadata["case_counts"].get(case), f"case_counts.{case}", 0)
        if expected != actual_counts[case]:
            raise ValueError("Stage A case count mismatch")
    if sum(actual_counts.values()) != nonsingular_rows:
        raise ValueError("Stage A case accounting mismatch")

    summary_buffer = StringIO(newline="")
    writer = csv.DictWriter(
        summary_buffer,
        fieldnames=STAGE_A_SUMMARY_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    for case in CASE_ORDER:
        c2, has3 = CASE_PARAMETERS[case]
        writer.writerow(
            {
                "case": case,
                "c2": c2,
                "has3": has3,
                "count": actual_counts[case],
                "share_nonsingular": (
                    f"{actual_counts[case] / nonsingular_rows:.12f}"
                    if nonsingular_rows
                    else "0.000000000000"
                ),
                "input_pairs": input_pairs,
                "nonsingular_rows": nonsingular_rows,
                "singular_excluded": singular_excluded,
                "seed": config.seed,
            }
        )
    return rows, metadata, summary_buffer.getvalue()


def _serialize_invariants(torsion: Any) -> str:
    return json.dumps([int(value) for value in torsion.invariants()], separators=(",", ":"))


def _serialize_generators(torsion: Any) -> str:
    coordinates = [
        [str(coordinate) for coordinate in generator.element()]
        for generator in torsion.gens()
    ]
    return json.dumps(coordinates, separators=(",", ":"))


def _validate_stage_b(
    input_dir: Path,
    config: Any,
    stage_a_rows: list[dict[str, str]],
    stage_a_metadata: dict[str, Any],
) -> tuple[str, int, dict[str, int]]:
    from sage.all import EllipticCurve, QQ
    from sage.version import version as sage_version

    if sage_version != "10.8":
        raise RuntimeError(f"Sage 10.8 is required, found {sage_version}")
    rows = _read_csv_exact(
        input_dir / "stage_b_reference_rows.csv",
        STAGE_B_COLUMNS,
    )
    selection = _read_json_object(
        input_dir / "stage_b_selection.json",
        STAGE_B_SELECTION_KEYS,
    )
    selection_seed = _require_plain_int(
        selection["seed"],
        "stage_b_selection.seed",
    )
    reference_total = _require_plain_int(
        selection["reference_total"],
        "stage_b_selection.reference_total",
        0,
    )
    if selection_seed != config.seed:
        raise ValueError("Stage B seed does not match configuration")
    if reference_total != config.reference_total:
        raise ValueError("Stage B reference_total does not match configuration")
    limit_per_case = selection["limit_per_case"]
    if limit_per_case is not None:
        limit_per_case = _require_plain_int(limit_per_case, "limit_per_case", 0)
    selected_total = _require_plain_int(selection["selected_total"], "selected_total", 0)
    if selected_total != len(rows) or selected_total > reference_total:
        raise ValueError("Stage B total cap or row count mismatch")

    for field in ("available_per_case", "selected_per_case"):
        value = selection[field]
        if not isinstance(value, dict) or list(value) != list(CASE_ORDER):
            raise ValueError(f"invalid Stage B {field}")
    available = {
        case: _require_plain_int(selection["available_per_case"].get(case), f"available.{case}", 0)
        for case in CASE_ORDER
    }
    expected_available = {
        case: _require_plain_int(stage_a_metadata["case_counts"].get(case), f"case_counts.{case}", 0)
        for case in CASE_ORDER
    }
    if available != expected_available:
        raise ValueError("Stage B available counts do not match Stage A")
    selected = {
        case: _require_plain_int(selection["selected_per_case"].get(case), f"selected.{case}", 0)
        for case in CASE_ORDER
    }
    if sum(selected.values()) != selected_total:
        raise ValueError("Stage B per-case selection accounting mismatch")
    if any(selected[case] > available[case] for case in CASE_ORDER):
        raise ValueError("Stage B selection exceeds available rows")

    expected_selected = allocate_reference_quota(
        available,
        reference_total,
        limit_per_case,
    )
    if selected != expected_selected:
        raise ValueError("Stage B quota does not match deterministic allocation")
    baskets = {
        case: [row for row in stage_a_rows if row["case"] == case]
        for case in CASE_ORDER
    }
    random_source = random.Random(selection_seed)
    expected_rows: list[dict[str, str]] = []
    for case in CASE_ORDER:
        candidates = list(baskets[case])
        random_source.shuffle(candidates)
        expected_rows.extend(candidates[: expected_selected[case]])
    expected_rows.sort(key=lambda row: (CASE_ORDER.index(row["case"]), row["sample_id"]))
    if [row["sample_id"] for row in rows] != [row["sample_id"] for row in expected_rows]:
        raise ValueError("Stage B seeded selection mismatch")

    stage_a_by_id = {row["sample_id"]: row for row in stage_a_rows}
    actual_selected = {case: 0 for case in CASE_ORDER}
    seen_ids: set[str] = set()
    normalized: list[dict[str, str]] = []
    for row in rows:
        sample_id = row["sample_id"]
        if SAMPLE_ID_PATTERN.fullmatch(sample_id) is None or sample_id in seen_ids:
            raise ValueError("duplicate or invalid Stage B sample_id")
        seen_ids.add(sample_id)
        source = stage_a_by_id.get(sample_id)
        if source is None:
            raise ValueError("Stage B row is not a member of Stage A")
        a = _parse_csv_int(row["a"], "a")
        b = _parse_csv_int(row["b"], "b")
        case = row["case"]
        if case not in CASE_ORDER:
            raise ValueError("invalid Stage B case")
        if source["a"] != str(a) or source["b"] != str(b) or source["case"] != case:
            raise ValueError("Stage B membership/case mismatch")
        torsion = EllipticCurve(QQ, [0, 0, 0, a, b]).torsion_subgroup()
        expected = {
            "sample_id": sample_id,
            "a": str(a),
            "b": str(b),
            "case": case,
            "torsion_order": str(int(torsion.order())),
            "torsion_invariants": _serialize_invariants(torsion),
            "generators": _serialize_generators(torsion),
        }
        if row != expected:
            raise ValueError(f"Sage reference mismatch: {sample_id}")
        actual_selected[case] += 1
        normalized.append(expected)
    if actual_selected != selected:
        raise ValueError("Stage B per-case row count mismatch")

    normalized.sort(key=lambda row: (CASE_ORDER.index(row["case"]), row["sample_id"]))
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=STAGE_B_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(normalized)
    return buffer.getvalue(), selected_total, selected


def _sage_has_three_torsion(a: int, b: int) -> bool:
    from sage.all import EllipticCurve, QQ

    invariants = EllipticCurve(QQ, [0, 0, 0, a, b]).torsion_subgroup().invariants()
    return any(int(value) % 3 == 0 for value in invariants)


def _generate_calibration(config: Any, per_class: int) -> list[tuple[int, int, int]]:
    if per_class < 1:
        raise ValueError("--calibration-per-class must be positive")
    random_source = random.Random(config.seed)
    used: set[tuple[int, int]] = set()
    negatives: list[tuple[int, int, int]] = []
    attempts = 0
    maximum_attempts = max(100000, per_class * 2000)
    while len(negatives) < per_class and attempts < maximum_attempts:
        attempts += 1
        a = random_source.randint(config.calibration_a_min, config.calibration_a_max)
        b = random_source.randint(config.calibration_b_min, config.calibration_b_max)
        pair = (a, b)
        if a == 0 or b == 0 or pair in used or not RationalCurve(a=a, b=b).is_nonsingular():
            continue
        if _sage_has_three_torsion(a, b):
            continue
        used.add(pair)
        negatives.append((a, b, 0))
    if len(negatives) != per_class:
        raise RuntimeError("could not generate the requested exact negative calibration rows")

    positives: list[tuple[int, int, int]] = []
    attempts = 0
    while len(positives) < per_class and attempts < maximum_attempts:
        attempts += 1
        u = random_source.randint(-20, 20)
        y = random_source.randint(-2000, 2000)
        if u == 0 or y == 0:
            continue
        x = 3 * u * u
        sign = -1 if random_source.getrandbits(1) == 0 else 1
        a = -3 * x * x + sign * 6 * u * y
        b = y * y - x * x * x - a * x
        pair = (a, b)
        if (
            a == 0
            or b == 0
            or pair in used
            or not config.calibration_a_min <= a <= config.calibration_a_max
            or not config.calibration_b_min <= b <= config.calibration_b_max
            or not RationalCurve(a=a, b=b).is_nonsingular()
        ):
            continue
        if not _sage_has_three_torsion(a, b):
            continue
        used.add(pair)
        positives.append((a, b, 1))
    if len(positives) != per_class:
        raise RuntimeError("could not generate the requested constructive positive calibration rows")
    return negatives + positives


def _build_release_classifier(build_dir: Path) -> Path:
    subprocess.run(
        [
            "cmake",
            "-S",
            str(REPOSITORY_ROOT / "cpp"),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--config", "Release"],
        check=True,
        capture_output=True,
        text=True,
    )
    for candidate in (build_dir / "phase_a", build_dir / "Release" / "phase_a"):
        if candidate.is_file():
            return candidate
    raise RuntimeError("C++ classifier executable was not produced")


def _run_cpp_classifier(
    rows: list[tuple[int, int, int]],
) -> list[int]:
    payload = "".join(
        f"calibration_{index:06d}\t{a}\t{b}\n"
        for index, (a, b, _) in enumerate(rows, start=1)
    )
    with tempfile.TemporaryDirectory(prefix="elliptic-diagnostics-cpp-") as temporary:
        executable = _build_release_classifier(Path(temporary))
        completed = subprocess.run(
            [str(executable)],
            input=payload,
            check=True,
            capture_output=True,
            text=True,
        )
    output = [line.split("\t") for line in completed.stdout.splitlines()]
    if len(output) != len(rows) or any(len(row) != 5 for row in output):
        raise RuntimeError("C++ classifier returned an invalid calibration schema")
    predictions: list[int] = []
    for index, fields in enumerate(output, start=1):
        if fields[0] != f"calibration_{index:06d}" or fields[2] not in {"0", "1"}:
            raise RuntimeError("C++ classifier changed calibration row order")
        predictions.append(int(fields[2]))
    return predictions


def _metric(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.12f}" if denominator else "0.000000000000"


def _calibration_result(
    backend: str,
    truth: list[int],
    predictions: list[int],
    fingerprint: str,
    config: Any,
    mismatches: int,
) -> dict[str, Any]:
    tp = sum(expected == actual == 1 for expected, actual in zip(truth, predictions, strict=True))
    fp = sum(expected == 0 and actual == 1 for expected, actual in zip(truth, predictions, strict=True))
    tn = sum(expected == actual == 0 for expected, actual in zip(truth, predictions, strict=True))
    fn = sum(expected == 1 and actual == 0 for expected, actual in zip(truth, predictions, strict=True))
    return {
        "backend": backend,
        "n": len(truth),
        "truth_positive": sum(truth),
        "truth_negative": len(truth) - sum(truth),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": _metric(tp, tp + fp),
        "recall": _metric(tp, tp + fn),
        "specificity": _metric(tn, tn + fp),
        "accuracy": _metric(tp + tn, len(truth)),
        "input_sha256": fingerprint,
        "seed": config.seed,
        "a_min": config.calibration_a_min,
        "a_max": config.calibration_a_max,
        "b_min": config.calibration_b_min,
        "b_max": config.calibration_b_max,
        "python_cpp_mismatches": mismatches,
    }


def _run_calibration(config: Any, per_class: int) -> tuple[str, int, str]:
    rows = _generate_calibration(config, per_class)
    truth = [truth_value for _, _, truth_value in rows]
    fingerprint_payload = "".join(
        f"{a},{b},{truth_value}\n" for a, b, truth_value in rows
    ).encode("ascii")
    fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()
    python_predictions = [
        int(classify_a1_a6(RationalCurve(a=a, b=b)).has_3_torsion_indicator)
        for a, b, _ in rows
    ]
    cpp_predictions = _run_cpp_classifier(rows)
    mismatches = sum(
        left != right
        for left, right in zip(python_predictions, cpp_predictions, strict=True)
    )
    if mismatches:
        raise ValueError(f"Python/C++ calibration mismatch count: {mismatches}")
    results = [
        _calibration_result(
            "python",
            truth,
            python_predictions,
            fingerprint,
            config,
            mismatches,
        ),
        _calibration_result(
            "cpp",
            truth,
            cpp_predictions,
            fingerprint,
            config,
            mismatches,
        ),
    ]
    if any(result["fp"] or result["fn"] for result in results):
        raise ValueError("calibration backend disagrees with exact Sage truth")
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CALIBRATION_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(results)
    return buffer.getvalue(), mismatches, fingerprint


def _validate_benchmark(path: Path) -> str:
    rows = _read_csv_exact(path, BENCHMARK_COLUMNS)
    if len(rows) != len(CANONICAL_BENCHMARK_ROWS):
        raise ValueError("benchmark_scaling.csv must contain four canonical rows")
    timing_columns = (
        "python_wall_median_s",
        "python_cpu_median_s",
        "cpp_wall_median_s",
        "cpp_cpu_median_s",
        "sage_wall_median_s",
        "sage_cpu_median_s",
    )
    mean_columns = (
        "two_torsion_candidates_mean",
        "three_torsion_candidates_mean",
        "exact_polynomial_checks_mean",
        "square_checks_mean",
    )
    ratio_contracts = (
        ("python_over_cpp_wall_x", "python_wall_median_s", "cpp_wall_median_s"),
        ("python_over_cpp_cpu_x", "python_cpu_median_s", "cpp_cpu_median_s"),
        ("sage_over_cpp_wall_x", "sage_wall_median_s", "cpp_wall_median_s"),
        ("sage_over_cpp_cpu_x", "sage_cpu_median_s", "cpp_cpu_median_s"),
        ("sage_over_python_wall_x", "sage_wall_median_s", "python_wall_median_s"),
        ("sage_over_python_cpu_x", "sage_cpu_median_s", "python_cpu_median_s"),
    )
    for row, (range_limit, seed, fingerprint) in zip(
        rows,
        CANONICAL_BENCHMARK_ROWS,
        strict=True,
    ):
        if (
            _parse_csv_int(row["range_limit"], "range_limit") != range_limit
            or _parse_csv_int(row["n"], "n") != 1000
            or _parse_csv_int(row["repeats"], "repeats") != 3
            or _parse_csv_int(row["seed"], "seed") != seed
            or row["input_sha256"] != fingerprint
        ):
            raise ValueError("benchmark canonical identity mismatch")
        values = {
            column: _parse_csv_float(row[column], column, 0.0)
            for column in (*timing_columns, *mean_columns)
        }
        if any(values[column] <= 0 for column in timing_columns):
            raise ValueError("benchmark timing must be positive")
        for ratio_column, numerator_column, denominator_column in ratio_contracts:
            ratio = _parse_csv_float(row[ratio_column], ratio_column, 0.0)
            expected_ratio = values[numerator_column] / values[denominator_column]
            if ratio <= 0 or not isclose(ratio, expected_ratio, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"benchmark ratio mismatch: {ratio_column}")
        if (
            _parse_csv_int(row["python_cpp_mismatches"], "python_cpp_mismatches") != 0
            or _parse_csv_int(row["python_sage_mismatches"], "python_sage_mismatches") != 0
        ):
            raise ValueError("benchmark backend mismatch count must be zero")
    return path.read_text(encoding="utf-8")


def _contains_sensitive_or_pathlike(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            _contains_sensitive_or_pathlike(key) or _contains_sensitive_or_pathlike(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_sensitive_or_pathlike(item) for item in value)
    if not isinstance(value, str):
        return False
    lowered = value.casefold()
    forbidden_fragments = (
        "hostname",
        "username",
        "git_dirty",
        "absolute_paths",
        "timestamp",
        str(Path.home()).casefold(),
        str(REPOSITORY_ROOT).casefold(),
    )
    return (
        any(fragment and fragment in lowered for fragment in forbidden_fragments)
        or Path(value).is_absolute()
        or re.match(r"^[A-Za-z]:[\\/]", value) is not None
        or value.startswith(("~/", "file:"))
    )


def _validate_environment(path: Path) -> str:
    environment = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(environment, dict) or set(environment) != ENVIRONMENT_KEYS:
        raise ValueError("invalid schema for environment.json")
    for key in (
        "os",
        "architecture",
        "cpu_model",
        "sage_version",
        "python_version",
        "compiler_version",
    ):
        if not isinstance(environment[key], str) or not environment[key].strip():
            raise ValueError(f"invalid environment field: {key}")
    logical_cores = environment["logical_cores"]
    memory_gib = environment["memory_gib"]
    if isinstance(logical_cores, bool) or not isinstance(logical_cores, int) or logical_cores <= 0:
        raise ValueError("invalid environment field: logical_cores")
    if (
        isinstance(memory_gib, bool)
        or not isinstance(memory_gib, (int, float))
        or not isfinite(float(memory_gib))
        or memory_gib <= 0
    ):
        raise ValueError("invalid environment field: memory_gib")
    if environment["sage_version"] != "10.8":
        raise ValueError("environment Sage version must be 10.8")
    if _contains_sensitive_or_pathlike(environment):
        raise ValueError("path-like value in environment.json")
    return path.read_text(encoding="utf-8")


def _validate_results_directory(results_dir: Path) -> dict[str, str]:
    benchmark_path = results_dir / "benchmark_scaling.csv"
    if not benchmark_path.is_file():
        raise ValueError("missing benchmark_scaling.csv")
    environment_path = results_dir / "environment.json"
    if not environment_path.is_file():
        raise ValueError("missing environment.json")
    unexpected = {path.name for path in results_dir.iterdir()} - FINAL_RESULT_FILES
    if unexpected:
        raise ValueError("unexpected files in results directory")
    return {
        "benchmark_scaling.csv": _validate_benchmark(benchmark_path),
        "environment.json": _validate_environment(environment_path),
    }


def _publish_result_set(results_dir: Path, publications: dict[str, str]) -> None:
    if set(publications) != FINAL_RESULT_FILES:
        raise ValueError("publication set must contain exactly five result files")
    parent = results_dir.parent
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{results_dir.name}.stage-", dir=parent))
    backup_dir = Path(tempfile.mkdtemp(prefix=f".{results_dir.name}.backup-", dir=parent))
    ordered_names = sorted(FINAL_RESULT_FILES)
    moved_old: list[str] = []
    committed_new: list[str] = []
    try:
        for filename in ordered_names:
            (stage_dir / filename).write_text(publications[filename], encoding="utf-8")
        for filename in ordered_names:
            current = results_dir / filename
            if current.is_file():
                current.replace(backup_dir / filename)
                moved_old.append(filename)
        for filename in ordered_names:
            (stage_dir / filename).replace(results_dir / filename)
            committed_new.append(filename)
    except BaseException:
        for filename in committed_new:
            current = results_dir / filename
            if current.exists():
                current.unlink()
        for filename in reversed(moved_old):
            backup = backup_dir / filename
            if backup.exists():
                backup.replace(results_dir / filename)
        raise
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
        shutil.rmtree(backup_dir, ignore_errors=True)


def _execute(arguments: argparse.Namespace) -> None:
    config = load_config(_root_relative(arguments.config))
    input_dir = _root_relative(arguments.input_dir)
    results_dir = _root_relative(arguments.results_dir)
    publications = _validate_results_directory(results_dir)
    stage_a_rows, stage_a_metadata, stage_a_text = _validate_stage_a(input_dir, config)
    stage_b_text, selected_total, selected_counts = _validate_stage_b(
        input_dir,
        config,
        stage_a_rows,
        stage_a_metadata,
    )
    calibration_text, mismatches, fingerprint = _run_calibration(
        config,
        arguments.calibration_per_class,
    )
    publications.update(
        {
            "stage_a_summary.csv": stage_a_text,
            "stage_b_reference.csv": stage_b_text,
            "calibration.csv": calibration_text,
        }
    )
    _publish_result_set(results_dir, publications)
    if {path.name for path in results_dir.iterdir()} != FINAL_RESULT_FILES:
        raise ValueError("required result file set is incomplete")
    print(f"Python/C++ mismatches: {mismatches}")
    print("Singular curves excluded: true")
    print("Sage reference rows valid: true")
    print("Required summaries present: true")
    print(
        "Calibration rows: "
        f"{arguments.calibration_per_class} positive, "
        f"{arguments.calibration_per_class} negative"
    )
    print(f"Calibration input SHA-256: {fingerprint}")
    print(
        "Sage reference selected: "
        f"{selected_total} "
        + " ".join(f"{case}={selected_counts[case]}" for case in CASE_ORDER)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", default="output")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--calibration-per-class", type=int, default=300)
    return parser


def _run_under_sage_if_needed() -> int | None:
    try:
        __import__("sage.all")
    except ModuleNotFoundError:
        sage = shutil.which("sage")
        if sage is None:
            print("Sage executable is required", file=sys.stderr)
            return 1
        completed = subprocess.run(
            [sage, "-python", str(Path(__file__).resolve()), *sys.argv[1:]],
            check=False,
            capture_output=True,
            text=True,
        )
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        return completed.returncode
    return None


def main() -> int:
    delegated = _run_under_sage_if_needed()
    if delegated is not None:
        return delegated
    try:
        _execute(_parser().parse_args())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
