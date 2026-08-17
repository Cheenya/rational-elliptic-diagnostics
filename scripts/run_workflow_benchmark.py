from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
from io import StringIO
import math
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from elliptic_diagnostics.classifier import classify_a1_a6
from elliptic_diagnostics.curve import RationalCurve
from elliptic_diagnostics.experiment import (
    CASE_ORDER,
    ExperimentConfig,
    allocate_reference_quota,
    load_config,
)


SCENARIO_ORDER = ("sage_all", "python_hybrid", "cpp_hybrid")
TIMING_COLUMNS = (
    "stage_a_wall_s",
    "selection_wall_s",
    "sage_wall_s",
    "serialization_wall_s",
    "total_wall_s",
)
RAW_COLUMNS = (
    "scenario",
    "repeat_index",
    "scenario_position",
    "n",
    "reference_total",
    "seed",
    "input_sha256",
    "reference_subset_sha256",
    *TIMING_COLUMNS,
    "diagnostic_count",
    "exact_count",
    "diagnostic_mismatches",
    "exact_subset_mismatches",
)
SUMMARY_COLUMNS = (
    "scenario",
    "n",
    "reference_total",
    "repeats",
    "seed",
    "input_sha256",
    "reference_subset_sha256",
    "stage_a_wall_median_s",
    "stage_a_wall_q1_s",
    "stage_a_wall_q3_s",
    "stage_a_wall_iqr_s",
    "selection_wall_median_s",
    "selection_wall_q1_s",
    "selection_wall_q3_s",
    "selection_wall_iqr_s",
    "sage_wall_median_s",
    "sage_wall_q1_s",
    "sage_wall_q3_s",
    "sage_wall_iqr_s",
    "serialization_wall_median_s",
    "serialization_wall_q1_s",
    "serialization_wall_q3_s",
    "serialization_wall_iqr_s",
    "total_wall_median_s",
    "total_wall_q1_s",
    "total_wall_q3_s",
    "total_wall_iqr_s",
    "diagnostic_count",
    "diagnostic_coverage",
    "exact_count",
    "exact_coverage",
    "diagnostic_mismatches",
    "exact_subset_mismatches",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _validated_sample_ids(rows: list[dict[str, object]]) -> list[str]:
    sample_ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("sample_id must be a nonempty string")
        if sample_id in seen:
            raise ValueError("sample_id must be unique")
        seen.add(sample_id)
        sample_ids.append(sample_id)
    return sample_ids


def hash_reference_subset(rows: list[dict[str, object]]) -> str:
    sample_ids = sorted(_validated_sample_ids(rows))
    payload = "".join(f"{sample_id}\n" for sample_id in sample_ids).encode("utf-8")
    return sha256(payload).hexdigest()


def select_reference_rows(
    rows: list[dict[str, object]],
    reference_total: int,
    seed: int,
) -> list[dict[str, object]]:
    if reference_total < 0:
        raise ValueError("reference_total must be nonnegative")
    if reference_total > len(rows):
        raise ValueError("reference_total cannot exceed the number of rows")

    _validated_sample_ids(rows)
    baskets: dict[str, list[dict[str, object]]] = {
        case: [] for case in CASE_ORDER
    }
    for row in rows:
        case = str(row["case"])
        if case not in baskets:
            raise ValueError(f"unknown diagnostic case: {case}")
        baskets[case].append(row)

    selected_per_case = allocate_reference_quota(
        {case: len(baskets[case]) for case in CASE_ORDER},
        reference_total,
    )
    random_source = random.Random(seed)
    selected: list[dict[str, object]] = []
    for case in CASE_ORDER:
        candidates = list(baskets[case])
        random_source.shuffle(candidates)
        selected.extend(candidates[: selected_per_case[case]])
    case_index = {case: index for index, case in enumerate(CASE_ORDER)}
    selected.sort(
        key=lambda row: (case_index[str(row["case"])], str(row["sample_id"]))
    )
    return [dict(row) for row in selected]


def _single_value(
    rows: list[dict[str, object]],
    column: str,
    converter,
):
    values = {converter(row[column]) for row in rows}
    if len(values) != 1:
        raise ValueError(f"{column} must be constant within a scenario")
    return values.pop()


def _timing_summary(values: list[float]) -> tuple[float, float, float, float]:
    median = float(statistics.median(values))
    if len(values) == 1:
        return median, median, median, 0.0
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return median, float(q1), float(q3), float(q3 - q1)


def _validate_sha256(value: str, column: str) -> None:
    if SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{column} must be 64 lowercase hexadecimal characters")


def summarize_runs(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not rows:
        raise ValueError("workflow benchmark rows must not be empty")

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        scenario = str(row["scenario"])
        if scenario not in SCENARIO_ORDER:
            raise ValueError(f"unknown workflow scenario: {scenario}")
        grouped[scenario].append(row)
    if set(grouped) != set(SCENARIO_ORDER):
        raise ValueError("all workflow scenarios are required")

    repeat_sets: list[set[int]] = []
    for scenario in SCENARIO_ORDER:
        scenario_rows = grouped[scenario]
        repeat_indices = [int(row["repeat_index"]) for row in scenario_rows]
        if len(repeat_indices) != len(set(repeat_indices)):
            raise ValueError(f"duplicate repeat_index for {scenario}")
        repeat_sets.append(set(repeat_indices))
    if any(repeats != repeat_sets[0] for repeats in repeat_sets[1:]):
        raise ValueError("workflow scenarios must contain the same repeats")
    if any(repeat_index < 1 for repeat_index in repeat_sets[0]):
        raise ValueError("repeat_index must be positive")

    for repeat_index in sorted(repeat_sets[0]):
        repeat_rows = [
            row for row in rows if int(row["repeat_index"]) == repeat_index
        ]
        positions = [int(row["scenario_position"]) for row in repeat_rows]
        if sorted(positions) != [1, 2, 3]:
            raise ValueError(
                "each repeat must use scenario positions 1, 2, and 3 exactly once"
            )
        actual_order = tuple(
            str(row["scenario"])
            for row in sorted(
                repeat_rows,
                key=lambda row: int(row["scenario_position"]),
            )
        )
        rotation = (repeat_index - 1) % len(SCENARIO_ORDER)
        expected_order = SCENARIO_ORDER[rotation:] + SCENARIO_ORDER[:rotation]
        if actual_order != expected_order:
            raise ValueError(
                f"repeat {repeat_index} does not use the expected cyclic scenario order"
            )

    n = _single_value(rows, "n", int)
    reference_total = _single_value(rows, "reference_total", int)
    seed = _single_value(rows, "seed", int)
    input_sha256 = _single_value(rows, "input_sha256", str)
    _validate_sha256(input_sha256, "input_sha256")
    if n <= 0:
        raise ValueError("n must be positive")
    if reference_total < 0 or reference_total > n:
        raise ValueError("reference_total must be between zero and n")

    reference_subset_hashes = {
        str(row["reference_subset_sha256"])
        for row in rows
    }
    if len(reference_subset_hashes) != 1:
        raise ValueError(
            "all workflow scenarios must use the same frozen reference subset"
        )
    reference_subset_sha256 = next(iter(reference_subset_hashes))
    _validate_sha256(reference_subset_sha256, "reference_subset_sha256")

    summary_rows: list[dict[str, object]] = []
    for scenario in SCENARIO_ORDER:
        scenario_rows = grouped[scenario]
        diagnostic_count = _single_value(
            scenario_rows, "diagnostic_count", int
        )
        exact_count = _single_value(scenario_rows, "exact_count", int)
        diagnostic_mismatches = _single_value(
            scenario_rows, "diagnostic_mismatches", int
        )
        exact_subset_mismatches = _single_value(
            scenario_rows, "exact_subset_mismatches", int
        )
        if diagnostic_mismatches != 0:
            raise ValueError("diagnostic_mismatches must be exactly zero")
        if exact_subset_mismatches != 0:
            raise ValueError("exact_subset_mismatches must be exactly zero")
        if diagnostic_count != n:
            raise ValueError(f"{scenario} must diagnose all input rows")
        expected_exact_count = n if scenario == "sage_all" else reference_total
        if exact_count != expected_exact_count:
            raise ValueError(
                f"{scenario} exact_count must equal {expected_exact_count}"
            )

        summary: dict[str, object] = {
            "scenario": scenario,
            "n": n,
            "reference_total": reference_total,
            "repeats": len(scenario_rows),
            "seed": seed,
            "input_sha256": input_sha256,
            "reference_subset_sha256": reference_subset_sha256,
        }
        for timing_column in TIMING_COLUMNS:
            values = [float(row[timing_column]) for row in scenario_rows]
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"{timing_column} must be finite")
            if timing_column == "total_wall_s" and any(
                value <= 0 for value in values
            ):
                raise ValueError("total_wall_s must be positive")
            if timing_column != "total_wall_s" and any(
                value < 0 for value in values
            ):
                raise ValueError(f"{timing_column} must be nonnegative")
            median, q1, q3, iqr = _timing_summary(values)
            prefix = timing_column.removesuffix("_s")
            summary[f"{prefix}_median_s"] = median
            summary[f"{prefix}_q1_s"] = q1
            summary[f"{prefix}_q3_s"] = q3
            summary[f"{prefix}_iqr_s"] = iqr
        summary.update(
            {
                "diagnostic_count": diagnostic_count,
                "diagnostic_coverage": diagnostic_count / n,
                "exact_count": exact_count,
                "exact_coverage": exact_count / n,
                "diagnostic_mismatches": diagnostic_mismatches,
                "exact_subset_mismatches": exact_subset_mismatches,
            }
        )
        summary_rows.append(summary)
    return summary_rows


def _root_relative(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _generate_input_rows(
    config: ExperimentConfig,
    count: int,
) -> list[dict[str, object]]:
    random_source = random.Random(config.seed)
    rows: list[dict[str, object]] = []
    while len(rows) < count:
        a = random_source.randint(config.a_min, config.a_max)
        b = random_source.randint(config.b_min, config.b_max)
        if not config.include_zero_coefficients and (a == 0 or b == 0):
            continue
        curve = RationalCurve(a=a, b=b)
        if not curve.is_nonsingular():
            continue
        rows.append(
            {
                "sample_id": f"sample_{len(rows) + 1:09d}",
                "a": a,
                "b": b,
            }
        )
    return rows


def _hash_input_rows(rows: list[dict[str, object]]) -> str:
    _validated_sample_ids(rows)
    payload = "".join(
        f'{row["sample_id"]}\t{int(row["a"])}\t{int(row["b"])}\n'
        for row in rows
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _build_cpp_classifier(build_dir: Path) -> Path:
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
    for executable in (build_dir / "phase_a", build_dir / "Release" / "phase_a"):
        if executable.is_file():
            return executable
    raise RuntimeError("C++ classifier executable was not produced")


def _python_diagnostics(
    input_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    diagnostics: list[dict[str, object]] = []
    for row in input_rows:
        result = classify_a1_a6(
            RationalCurve(a=int(row["a"]), b=int(row["b"]))
        )
        diagnostics.append(
            {
                "sample_id": row["sample_id"],
                "a": row["a"],
                "b": row["b"],
                "c2": result.c2,
                "has3": result.has_3_torsion_indicator,
                "case": result.case,
            }
        )
    return diagnostics


def _cpp_diagnostics(
    executable: Path,
    input_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    classifier_input = "".join(
        f'{row["sample_id"]}\t{int(row["a"])}\t{int(row["b"])}\n'
        for row in input_rows
    )
    completed = subprocess.run(
        [str(executable)],
        input=classifier_input,
        check=True,
        capture_output=True,
        text=True,
    )
    output_lines = completed.stdout.splitlines()
    if len(output_lines) != len(input_rows):
        raise RuntimeError("C++ classifier returned an invalid row count")

    diagnostics: list[dict[str, object]] = []
    for input_row, output_line in zip(input_rows, output_lines, strict=True):
        fields = output_line.split("\t")
        if len(fields) != 5 or fields[0] != input_row["sample_id"]:
            raise RuntimeError("C++ classifier returned an invalid schema or row order")
        c2_text, has3_text, case = fields[1:4]
        if (
            c2_text not in {"0", "1", "3"}
            or has3_text not in {"0", "1"}
            or case not in CASE_ORDER
        ):
            raise RuntimeError("C++ classifier returned invalid diagnostic values")
        diagnostics.append(
            {
                "sample_id": input_row["sample_id"],
                "a": input_row["a"],
                "b": input_row["b"],
                "c2": int(c2_text),
                "has3": has3_text == "1",
                "case": case,
            }
        )
    return diagnostics


def _sage_exact_invariants(
    rows: list[dict[str, object]],
) -> dict[str, tuple[int, ...]]:
    from sage.all import EllipticCurve, QQ

    exact: dict[str, tuple[int, ...]] = {}
    for row in rows:
        torsion = EllipticCurve(
            QQ,
            [0, 0, 0, int(row["a"]), int(row["b"])],
        ).torsion_subgroup()
        exact[str(row["sample_id"])] = tuple(
            int(value) for value in torsion.invariants()
        )
    return exact


def _case_from_exact_diagnostics(c2: int, has3: bool) -> str:
    cases = {
        (0, True): "A1",
        (1, True): "A2",
        (3, True): "A3",
        (0, False): "A4",
        (1, False): "A5",
        (3, False): "A6",
    }
    try:
        return cases[(c2, has3)]
    except KeyError as error:
        raise RuntimeError(
            f"unexpected exact torsion diagnostics: c2={c2}, has3={has3}"
        ) from error


def _sage_diagnostics(
    input_rows: list[dict[str, object]],
    exact: dict[str, tuple[int, ...]],
) -> list[dict[str, object]]:
    diagnostics: list[dict[str, object]] = []
    for row in input_rows:
        invariants = exact[str(row["sample_id"])]
        c2 = math.prod(math.gcd(value, 2) for value in invariants) - 1
        has3 = any(value % 3 == 0 for value in invariants)
        diagnostics.append(
            {
                "sample_id": row["sample_id"],
                "a": row["a"],
                "b": row["b"],
                "c2": c2,
                "has3": has3,
                "case": _case_from_exact_diagnostics(c2, has3),
            }
        )
    return diagnostics


def _diagnostic_map(
    rows: list[dict[str, object]],
) -> dict[str, tuple[int, bool, str]]:
    return {
        str(row["sample_id"]): (
            int(row["c2"]),
            bool(row["has3"]),
            str(row["case"]),
        )
        for row in rows
    }


def _hash_scenario_outputs(
    diagnostic_rows: list[dict[str, object]],
    exact: dict[str, tuple[int, ...]],
) -> str:
    diagnostic_payload = "".join(
        f'{row["sample_id"]}\t{int(row["c2"])}\t'
        f'{int(bool(row["has3"]))}\t{row["case"]}\n'
        for row in sorted(diagnostic_rows, key=lambda item: str(item["sample_id"]))
    )
    exact_payload = "".join(
        f'{sample_id}\t{",".join(str(value) for value in exact[sample_id])}\n'
        for sample_id in sorted(exact)
    )
    return sha256(f"{diagnostic_payload}--exact--\n{exact_payload}".encode("utf-8")).hexdigest()


def _mapping_mismatches(
    expected: dict[str, object],
    actual: dict[str, object],
    sample_ids: tuple[str, ...],
) -> int:
    return sum(expected.get(sample_id) != actual.get(sample_id) for sample_id in sample_ids)


def _run_scenario(
    scenario: str,
    repeat_index: int,
    scenario_position: int,
    executable: Path,
    input_rows: list[dict[str, object]],
    reference_total: int,
    seed: int,
    canonical_input_sha256: str,
    baseline_diagnostics: dict[str, tuple[int, bool, str]] | None,
    baseline_exact: dict[str, tuple[int, ...]] | None,
    frozen_subset_sha256: str | None,
) -> tuple[
    dict[str, object],
    dict[str, tuple[int, bool, str]],
    dict[str, tuple[int, ...]],
    str,
]:
    if baseline_diagnostics is None and scenario != "sage_all":
        raise RuntimeError("the first workflow scenario must establish Sage baseline")

    total_start = time.perf_counter_ns()
    stage_a_wall_s = 0.0
    if scenario == "sage_all":
        sage_start = time.perf_counter_ns()
        exact = _sage_exact_invariants(input_rows)
        diagnostic_rows = _sage_diagnostics(input_rows, exact)
        sage_wall_s = (time.perf_counter_ns() - sage_start) / 1_000_000_000

        selection_start = time.perf_counter_ns()
        selected_rows = select_reference_rows(
            diagnostic_rows,
            reference_total,
            seed,
        )
        selection_wall_s = (
            time.perf_counter_ns() - selection_start
        ) / 1_000_000_000
    else:
        stage_a_start = time.perf_counter_ns()
        if scenario == "python_hybrid":
            diagnostic_rows = _python_diagnostics(input_rows)
        elif scenario == "cpp_hybrid":
            diagnostic_rows = _cpp_diagnostics(executable, input_rows)
        else:
            raise ValueError(f"unknown workflow scenario: {scenario}")
        stage_a_wall_s = (
            time.perf_counter_ns() - stage_a_start
        ) / 1_000_000_000

        selection_start = time.perf_counter_ns()
        selected_rows = select_reference_rows(
            diagnostic_rows,
            reference_total,
            seed,
        )
        selection_wall_s = (
            time.perf_counter_ns() - selection_start
        ) / 1_000_000_000

        sage_start = time.perf_counter_ns()
        exact = _sage_exact_invariants(selected_rows)
        sage_wall_s = (time.perf_counter_ns() - sage_start) / 1_000_000_000

    serialization_start = time.perf_counter_ns()
    scenario_input_sha256 = _hash_input_rows(input_rows)
    reference_subset_sha256 = hash_reference_subset(selected_rows)
    _hash_scenario_outputs(diagnostic_rows, exact)
    serialization_wall_s = (
        time.perf_counter_ns() - serialization_start
    ) / 1_000_000_000

    if scenario_input_sha256 != canonical_input_sha256:
        raise RuntimeError("workflow input hash changed between scenarios")
    if (
        frozen_subset_sha256 is not None
        and reference_subset_sha256 != frozen_subset_sha256
    ):
        raise RuntimeError("workflow scenarios selected different reference subsets")

    actual_diagnostics = _diagnostic_map(diagnostic_rows)
    selected_ids = tuple(str(row["sample_id"]) for row in selected_rows)
    if baseline_diagnostics is None or baseline_exact is None:
        diagnostic_mismatches = 0
        exact_subset_mismatches = 0
    else:
        diagnostic_mismatches = _mapping_mismatches(
            baseline_diagnostics,
            actual_diagnostics,
            tuple(sorted(baseline_diagnostics)),
        )
        exact_subset_mismatches = _mapping_mismatches(
            baseline_exact,
            exact,
            selected_ids,
        )
    if diagnostic_mismatches or exact_subset_mismatches:
        raise RuntimeError(
            "workflow scenario disagrees with the first Sage-all baseline"
        )

    total_wall_s = (time.perf_counter_ns() - total_start) / 1_000_000_000
    raw_row: dict[str, object] = {
        "scenario": scenario,
        "repeat_index": repeat_index,
        "scenario_position": scenario_position,
        "n": len(input_rows),
        "reference_total": reference_total,
        "seed": seed,
        "input_sha256": canonical_input_sha256,
        "reference_subset_sha256": reference_subset_sha256,
        "stage_a_wall_s": stage_a_wall_s,
        "selection_wall_s": selection_wall_s,
        "sage_wall_s": sage_wall_s,
        "serialization_wall_s": serialization_wall_s,
        "total_wall_s": total_wall_s,
        "diagnostic_count": len(actual_diagnostics),
        "exact_count": len(exact),
        "diagnostic_mismatches": diagnostic_mismatches,
        "exact_subset_mismatches": exact_subset_mismatches,
    }
    return raw_row, actual_diagnostics, exact, reference_subset_sha256


def _warm_up(
    executable: Path,
    input_rows: list[dict[str, object]],
) -> None:
    warmup_rows = input_rows[:1]
    _python_diagnostics(warmup_rows)
    _cpp_diagnostics(executable, warmup_rows)
    _sage_exact_invariants(warmup_rows)


def _csv_text(
    rows: list[dict[str, object]],
    columns: tuple[str, ...],
) -> str:
    formatted_rows: list[dict[str, object]] = []
    for row in rows:
        formatted_rows.append(
            {
                column: (
                    f"{float(row[column]):.12f}"
                    if column.endswith("_s")
                    else row[column]
                )
                for column in columns
            }
        )
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(formatted_rows)
    return buffer.getvalue()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            handle.write(text)
            temporary_path = Path(handle.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _validate_cli_arguments(
    parser: argparse.ArgumentParser,
    n: int,
    k: int,
    repeats: int,
) -> None:
    if n <= 0:
        parser.error("--n must be positive")
    if k < 0 or k > n:
        parser.error("--k must be between zero and --n")
    if repeats <= 0:
        parser.error("--repeats must be positive")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()
    _validate_cli_arguments(
        parser,
        arguments.n,
        arguments.k,
        arguments.repeats,
    )

    config = load_config(_root_relative(arguments.config))
    input_rows = _generate_input_rows(config, arguments.n)
    canonical_input_sha256 = _hash_input_rows(input_rows)
    raw_rows: list[dict[str, object]] = []
    baseline_diagnostics: dict[str, tuple[int, bool, str]] | None = None
    baseline_exact: dict[str, tuple[int, ...]] | None = None
    frozen_subset_sha256: str | None = None

    with tempfile.TemporaryDirectory(
        prefix="rational-elliptic-workflow-benchmark-"
    ) as temporary_directory:
        executable = _build_cpp_classifier(Path(temporary_directory))
        _warm_up(executable, input_rows)
        for repeat_index in range(1, arguments.repeats + 1):
            rotation = (repeat_index - 1) % len(SCENARIO_ORDER)
            scenario_order = SCENARIO_ORDER[rotation:] + SCENARIO_ORDER[:rotation]
            for scenario_position, scenario in enumerate(scenario_order, start=1):
                (
                    raw_row,
                    actual_diagnostics,
                    actual_exact,
                    actual_subset_sha256,
                ) = _run_scenario(
                    scenario,
                    repeat_index,
                    scenario_position,
                    executable,
                    input_rows,
                    arguments.k,
                    config.seed,
                    canonical_input_sha256,
                    baseline_diagnostics,
                    baseline_exact,
                    frozen_subset_sha256,
                )
                if baseline_diagnostics is None:
                    baseline_diagnostics = actual_diagnostics
                    baseline_exact = actual_exact
                    frozen_subset_sha256 = actual_subset_sha256
                raw_rows.append(raw_row)

    summary_rows = summarize_runs(raw_rows)
    runs_text = _csv_text(raw_rows, RAW_COLUMNS)
    summary_text = _csv_text(summary_rows, SUMMARY_COLUMNS)
    output_dir = _root_relative(arguments.output_dir)
    _write_text_atomic(output_dir / "workflow_benchmark_runs.csv", runs_text)
    _write_text_atomic(output_dir / "workflow_benchmark_summary.csv", summary_text)


if __name__ == "__main__":
    main()
