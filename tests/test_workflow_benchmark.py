from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
import math
from pathlib import Path
import re
import shutil
import subprocess

import pytest

import scripts.run_workflow_benchmark as workflow_benchmark
from scripts.run_workflow_benchmark import (
    RAW_COLUMNS,
    SCENARIO_ORDER,
    SUMMARY_COLUMNS,
    select_reference_rows,
    summarize_runs,
)


EXPECTED_RAW_COLUMNS = (
    "scenario",
    "repeat_index",
    "scenario_position",
    "n",
    "reference_total",
    "seed",
    "input_sha256",
    "reference_subset_sha256",
    "stage_a_wall_s",
    "selection_wall_s",
    "sage_wall_s",
    "serialization_wall_s",
    "total_wall_s",
    "diagnostic_count",
    "exact_count",
    "diagnostic_mismatches",
    "exact_subset_mismatches",
)
EXPECTED_SUMMARY_COLUMNS = (
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


def _raw_rows(
    *,
    repeats: int = 5,
    reference_subset_hash: str | None = None,
    sage_subset_hash: str | None = None,
    cpp_subset_hash: str | None = None,
) -> list[dict[str, object]]:
    input_hash = "a" * 64
    frozen_subset_hash = reference_subset_hash or "b" * 64
    sage_hash = sage_subset_hash or frozen_subset_hash
    cpp_hash = cpp_subset_hash or frozen_subset_hash
    total_times = {
        "sage_all": (10.0, 12.0, 14.0, 16.0, 18.0),
        "python_hybrid": (4.0, 5.0, 6.0, 7.0, 8.0),
        "cpp_hybrid": (2.0, 3.0, 4.0, 5.0, 6.0),
    }
    rows: list[dict[str, object]] = []
    for repeat_index in range(1, repeats + 1):
        rotation = (repeat_index - 1) % len(SCENARIO_ORDER)
        scenario_order = SCENARIO_ORDER[rotation:] + SCENARIO_ORDER[:rotation]
        for scenario_position, scenario in enumerate(scenario_order, start=1):
            total_wall = total_times[scenario][repeat_index - 1]
            is_sage_all = scenario == "sage_all"
            rows.append(
                {
                    "scenario": scenario,
                    "repeat_index": repeat_index,
                    "scenario_position": scenario_position,
                    "n": "200",
                    "reference_total": "20",
                    "seed": "20260220",
                    "input_sha256": input_hash,
                    "reference_subset_sha256": (
                        sage_hash
                        if is_sage_all
                        else cpp_hash
                        if scenario == "cpp_hybrid"
                        else frozen_subset_hash
                    ),
                    "stage_a_wall_s": total_wall / 10,
                    "selection_wall_s": 0.1,
                    "sage_wall_s": total_wall / 2,
                    "serialization_wall_s": 0.2,
                    "total_wall_s": total_wall,
                    "diagnostic_count": "200",
                    "exact_count": "200" if is_sage_all else "20",
                    "diagnostic_mismatches": "0",
                    "exact_subset_mismatches": "0",
                }
            )
    return rows


def test_reference_selection_is_seeded_stable_and_preserves_input_order() -> None:
    rows = [
        {"sample_id": f"sample_{index:03d}", "case": "A4", "value": index}
        for index in range(8)
    ]
    original = deepcopy(rows)

    first = select_reference_rows(rows, reference_total=3, seed=20260220)
    second = select_reference_rows(rows, reference_total=3, seed=20260220)

    assert [row["sample_id"] for row in first] == [
        "sample_000",
        "sample_003",
        "sample_006",
    ]
    assert second == first
    assert rows == original
    source_by_id = {row["sample_id"]: row for row in rows}
    assert all(
        selected is not source_by_id[selected["sample_id"]]
        for selected in first
    )


def test_reference_selection_reproduces_canonical_case_stratification() -> None:
    rows = [
        {"sample_id": f"sample_{index:09d}", "case": "A4"}
        for index in range(199_904)
    ]
    rows.extend(
        {"sample_id": f"sample_{index:09d}", "case": "A5"}
        for index in range(199_904, 200_000)
    )

    selected = select_reference_rows(
        rows,
        reference_total=2_000,
        seed=20260220,
    )

    assert Counter(row["case"] for row in selected) == {
        "A4": 1_904,
        "A5": 96,
    }
    selected_keys = [(str(row["case"]), str(row["sample_id"])) for row in selected]
    assert selected_keys == sorted(selected_keys)
    assert all(
        selected_row
        is not rows[
            int(str(selected_row["sample_id"]).removeprefix("sample_"))
        ]
        for selected_row in selected
    )


@pytest.mark.parametrize("bad_sample_id", ["", "   ", None])
def test_reference_selection_rejects_empty_sample_ids(
    bad_sample_id: object,
) -> None:
    rows = [
        {"sample_id": "sample_001", "case": "A4"},
        {"sample_id": bad_sample_id, "case": "A5"},
    ]

    with pytest.raises(ValueError, match="sample_id must be a nonempty string"):
        select_reference_rows(rows, reference_total=1, seed=20260220)


def test_reference_selection_rejects_duplicate_sample_ids() -> None:
    rows = [
        {"sample_id": "sample_001", "case": "A4"},
        {"sample_id": "sample_001", "case": "A5"},
    ]

    with pytest.raises(ValueError, match="sample_id must be unique"):
        select_reference_rows(rows, reference_total=1, seed=20260220)


def test_reference_subset_hash_is_order_stable_and_sensitive_to_ids() -> None:
    first_subset = [
        {"sample_id": "sample_002"},
        {"sample_id": "sample_001"},
    ]
    changed_subset = [
        {"sample_id": "sample_001"},
        {"sample_id": "sample_003"},
    ]

    first_hash = workflow_benchmark.hash_reference_subset(first_subset)

    assert first_hash == "16e45ed75c9c642dd8455a91354061eebda2010a91a2734195542ade218a8763"
    assert workflow_benchmark.hash_reference_subset(list(reversed(first_subset))) == first_hash
    assert workflow_benchmark.hash_reference_subset(changed_subset) != first_hash


def test_summary_rejects_different_python_and_cpp_reference_subsets() -> None:
    rows = _raw_rows(cpp_subset_hash="c" * 64)

    with pytest.raises(ValueError, match="same frozen reference subset"):
        summarize_runs(rows)


def test_summary_rejects_a_different_sage_projection_subset() -> None:
    rows = _raw_rows(sage_subset_hash="d" * 64)

    with pytest.raises(ValueError, match="same frozen reference subset"):
        summarize_runs(rows)


def test_summary_rejects_different_hashes_from_real_reference_subsets() -> None:
    first_hash = workflow_benchmark.hash_reference_subset(
        [{"sample_id": "sample_001"}, {"sample_id": "sample_002"}]
    )
    second_hash = workflow_benchmark.hash_reference_subset(
        [{"sample_id": "sample_001"}, {"sample_id": "sample_003"}]
    )
    rows = _raw_rows(
        reference_subset_hash=first_hash,
        cpp_subset_hash=second_hash,
    )

    with pytest.raises(ValueError, match="same frozen reference subset"):
        summarize_runs(rows)


@pytest.mark.parametrize(
    ("column", "bad_hash"),
    [
        ("input_sha256", "A" * 64),
        ("input_sha256", "g" * 64),
        ("input_sha256", "a" * 63),
        ("reference_subset_sha256", "B" * 64),
        ("reference_subset_sha256", "z" * 64),
        ("reference_subset_sha256", "b" * 65),
    ],
)
def test_summary_rejects_noncanonical_sha256(
    column: str,
    bad_hash: str,
) -> None:
    rows = _raw_rows(repeats=1)
    for row in rows:
        row[column] = bad_hash

    with pytest.raises(ValueError, match=f"{column} must be 64 lowercase hex"):
        summarize_runs(rows)


@pytest.mark.parametrize(
    "column",
    ["diagnostic_mismatches", "exact_subset_mismatches"],
)
@pytest.mark.parametrize("bad_value", [-1, 1])
def test_summary_rejects_every_nonzero_mismatch_count(
    column: str,
    bad_value: int,
) -> None:
    rows = _raw_rows(repeats=1)
    for row in rows:
        row[column] = bad_value

    with pytest.raises(ValueError, match=f"{column} must be exactly zero"):
        summarize_runs(rows)


@pytest.mark.parametrize(
    "timing_column",
    [
        "stage_a_wall_s",
        "selection_wall_s",
        "sage_wall_s",
        "serialization_wall_s",
        "total_wall_s",
    ],
)
@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(math.nan, id="nan"),
        pytest.param(math.inf, id="positive-infinity"),
        pytest.param(-math.inf, id="negative-infinity"),
    ],
)
def test_summary_rejects_nonfinite_timings(
    timing_column: str,
    bad_value: float,
) -> None:
    rows = _raw_rows(repeats=1)
    for row in rows:
        row[timing_column] = bad_value

    with pytest.raises(ValueError, match=f"{timing_column} must be finite"):
        summarize_runs(rows)


@pytest.mark.parametrize(
    "timing_column",
    [
        "stage_a_wall_s",
        "selection_wall_s",
        "sage_wall_s",
        "serialization_wall_s",
    ],
)
def test_summary_rejects_negative_component_timings(timing_column: str) -> None:
    rows = _raw_rows(repeats=1)
    for row in rows:
        row[timing_column] = -0.001

    with pytest.raises(ValueError, match=f"{timing_column} must be nonnegative"):
        summarize_runs(rows)


def test_summary_rejects_zero_total_wall_time() -> None:
    rows = _raw_rows(repeats=1)
    for row in rows:
        row["total_wall_s"] = 0.0

    with pytest.raises(ValueError, match="total_wall_s must be positive"):
        summarize_runs(rows)


def test_summary_rejects_duplicate_scenario_positions_within_a_repeat() -> None:
    rows = _raw_rows(repeats=2)
    for row in rows:
        if row["repeat_index"] == 1 and row["scenario"] == "cpp_hybrid":
            row["scenario_position"] = 1

    with pytest.raises(ValueError, match="positions 1, 2, and 3 exactly once"):
        summarize_runs(rows)


def test_summary_rejects_noncyclic_scenario_order() -> None:
    rows = _raw_rows(repeats=2)
    for row in rows:
        if row["repeat_index"] != 2:
            continue
        if row["scenario"] == "sage_all":
            row["scenario_position"] = 2
        elif row["scenario"] == "cpp_hybrid":
            row["scenario_position"] = 3

    with pytest.raises(ValueError, match="cyclic scenario order"):
        summarize_runs(rows)


def test_summary_computes_inclusive_quartiles_and_exact_coverage() -> None:
    summary = summarize_runs(_raw_rows())

    assert [row["scenario"] for row in summary] == list(SCENARIO_ORDER)
    assert [row["total_wall_median_s"] for row in summary] == [14.0, 6.0, 4.0]
    assert [row["total_wall_q1_s"] for row in summary] == [12.0, 5.0, 3.0]
    assert [row["total_wall_q3_s"] for row in summary] == [16.0, 7.0, 5.0]
    assert [row["total_wall_iqr_s"] for row in summary] == [4.0, 2.0, 2.0]
    assert [row["diagnostic_coverage"] for row in summary] == [1.0, 1.0, 1.0]
    assert [row["exact_coverage"] for row in summary] == [1.0, 0.1, 0.1]
    assert [row["diagnostic_count"] for row in summary] == [200, 200, 200]
    assert [row["exact_count"] for row in summary] == [200, 20, 20]
    assert [row["repeats"] for row in summary] == [5, 5, 5]


def test_summary_uses_zero_iqr_for_a_single_repeat() -> None:
    summary = summarize_runs(_raw_rows(repeats=1))

    assert [row["total_wall_median_s"] for row in summary] == [10.0, 4.0, 2.0]
    assert [row["total_wall_iqr_s"] for row in summary] == [0.0, 0.0, 0.0]


def test_workflow_csv_column_order_is_stable() -> None:
    assert RAW_COLUMNS == EXPECTED_RAW_COLUMNS
    assert SUMMARY_COLUMNS == EXPECTED_SUMMARY_COLUMNS
    assert tuple(summarize_runs(_raw_rows(repeats=1))[0]) == SUMMARY_COLUMNS


def test_workflow_cli_runs_three_real_scenarios_with_rotated_order(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    sage_executable = shutil.which("sage")
    assert sage_executable is not None
    output_dir = tmp_path / "workflow-output"

    completed = subprocess.run(
        [
            sage_executable,
            "-python",
            str(repository_root / "scripts" / "run_workflow_benchmark.py"),
            "--config",
            str(repository_root / "configs" / "conference.yml"),
            "--n",
            "6",
            "--k",
            "3",
            "--repeats",
            "3",
            "--output-dir",
            str(output_dir),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert output_dir.is_dir()
    assert {path.name for path in output_dir.iterdir()} == {
        "workflow_benchmark_runs.csv",
        "workflow_benchmark_summary.csv",
    }

    runs_path = output_dir / "workflow_benchmark_runs.csv"
    summary_path = output_dir / "workflow_benchmark_summary.csv"
    assert b"\r\n" not in runs_path.read_bytes()
    assert b"\r\n" not in summary_path.read_bytes()
    with runs_path.open(newline="", encoding="utf-8") as handle:
        runs_reader = csv.DictReader(handle)
        run_rows = list(runs_reader)
        assert runs_reader.fieldnames == list(RAW_COLUMNS)
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary_reader = csv.DictReader(handle)
        summary_rows = list(summary_reader)
        assert summary_reader.fieldnames == list(SUMMARY_COLUMNS)

    assert [
        (int(row["repeat_index"]), int(row["scenario_position"]), row["scenario"])
        for row in run_rows
    ] == [
        (1, 1, "sage_all"),
        (1, 2, "python_hybrid"),
        (1, 3, "cpp_hybrid"),
        (2, 1, "python_hybrid"),
        (2, 2, "cpp_hybrid"),
        (2, 3, "sage_all"),
        (3, 1, "cpp_hybrid"),
        (3, 2, "sage_all"),
        (3, 3, "python_hybrid"),
    ]
    assert len(run_rows) == 9
    assert {row["input_sha256"] for row in run_rows} == {
        run_rows[0]["input_sha256"]
    }
    assert {row["reference_subset_sha256"] for row in run_rows} == {
        run_rows[0]["reference_subset_sha256"]
    }
    assert re.fullmatch(r"[0-9a-f]{64}", run_rows[0]["input_sha256"])
    assert re.fullmatch(
        r"[0-9a-f]{64}", run_rows[0]["reference_subset_sha256"]
    )
    assert all(row["n"] == "6" for row in run_rows)
    assert all(row["reference_total"] == "3" for row in run_rows)
    assert all(row["seed"] == "20260220" for row in run_rows)
    assert all(row["diagnostic_count"] == "6" for row in run_rows)
    assert all(
        int(row["exact_count"])
        == (6 if row["scenario"] == "sage_all" else 3)
        for row in run_rows
    )
    assert all(row["diagnostic_mismatches"] == "0" for row in run_rows)
    assert all(row["exact_subset_mismatches"] == "0" for row in run_rows)
    raw_timing_columns = [column for column in RAW_COLUMNS if column.endswith("_s")]
    assert all(
        re.fullmatch(r"[0-9]+\.[0-9]{12}", row[column])
        for row in run_rows
        for column in raw_timing_columns
    )
    assert all(float(row["total_wall_s"]) > 0 for row in run_rows)
    assert all(
        float(row["stage_a_wall_s"]) == 0
        for row in run_rows
        if row["scenario"] == "sage_all"
    )

    assert [row["scenario"] for row in summary_rows] == list(SCENARIO_ORDER)
    assert len(summary_rows) == 3
    assert {row["input_sha256"] for row in summary_rows} == {
        run_rows[0]["input_sha256"]
    }
    assert {row["reference_subset_sha256"] for row in summary_rows} == {
        run_rows[0]["reference_subset_sha256"]
    }
    assert [row["repeats"] for row in summary_rows] == ["3", "3", "3"]
    assert [row["diagnostic_count"] for row in summary_rows] == ["6", "6", "6"]
    assert [row["exact_count"] for row in summary_rows] == ["6", "3", "3"]
    assert [row["diagnostic_coverage"] for row in summary_rows] == [
        "1.0",
        "1.0",
        "1.0",
    ]
    assert [row["exact_coverage"] for row in summary_rows] == [
        "1.0",
        "0.5",
        "0.5",
    ]
    assert all(row["diagnostic_mismatches"] == "0" for row in summary_rows)
    assert all(row["exact_subset_mismatches"] == "0" for row in summary_rows)
    summary_timing_columns = [
        column for column in SUMMARY_COLUMNS if column.endswith("_s")
    ]
    assert all(
        re.fullmatch(r"[0-9]+\.[0-9]{12}", row[column])
        for row in summary_rows
        for column in summary_timing_columns
    )
