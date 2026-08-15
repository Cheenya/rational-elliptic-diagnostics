from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from elliptic_diagnostics.experiment import (
    CASE_ORDER,
    allocate_reference_quota,
    generate_candidate_pairs,
    load_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "conference.yml"


def _valid_config_text() -> str:
    return """\
seed: 20260220
a_min: -10000
a_max: 10000
b_min: -10000
b_max: 10000
sample_size: 200000
reference_total: 2000
include_zero_coefficients: false
calibration_a_min: -1000000
calibration_a_max: 1000000
calibration_b_min: -1000000
calibration_b_max: 1000000
"""


def _config_text_with(**overrides: int | bool) -> str:
    values: dict[str, str] = {}
    for line in _valid_config_text().splitlines():
        key, value = line.split(":", maxsplit=1)
        values[key] = value.strip()
    for key, value in overrides.items():
        values[key] = str(value).lower() if isinstance(value, bool) else str(value)
    return "".join(f"{key}: {value}\n" for key, value in values.items())


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (
            "seed: 20260220\nseed: 9\n",
            "duplicate configuration key: seed",
        ),
        (
            "",
            "missing configuration key: seed",
        ),
        (
            "seed: 20260220\nunexpected: 9\n",
            "unknown configuration key: unexpected",
        ),
    ],
)
def test_config_loader_rejects_duplicate_missing_and_unknown_keys(
    tmp_path: Path,
    replacement: str,
    message: str,
) -> None:
    config_path = tmp_path / "conference.yml"
    lines = _valid_config_text().splitlines(keepends=True)
    config_path.write_text(replacement + "".join(lines[1:]), encoding="utf-8")

    with pytest.raises(ValueError, match=f"^{message}$"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"a_min": 2, "a_max": 1}, "a_min must be less than or equal to a_max"),
        ({"b_min": 2, "b_max": 1}, "b_min must be less than or equal to b_max"),
        (
            {"calibration_a_min": 2, "calibration_a_max": 1},
            "calibration_a_min must be less than or equal to calibration_a_max",
        ),
        (
            {"calibration_b_min": 2, "calibration_b_max": 1},
            "calibration_b_min must be less than or equal to calibration_b_max",
        ),
        ({"sample_size": -1}, "sample_size must be nonnegative"),
        ({"reference_total": -1}, "reference_total must be nonnegative"),
        (
            {"a_min": 0, "a_max": 0},
            "a range cannot produce a nonzero coefficient",
        ),
        (
            {"b_min": 0, "b_max": 0},
            "b range cannot produce a nonzero coefficient",
        ),
    ],
)
def test_config_loader_rejects_semantically_invalid_experiment_ranges(
    tmp_path: Path,
    overrides: dict[str, int | bool],
    message: str,
) -> None:
    config_path = tmp_path / "conference.yml"
    config_path.write_text(_config_text_with(**overrides), encoding="utf-8")

    with pytest.raises(ValueError, match=f"^{message}$"):
        load_config(config_path)


def test_candidate_generation_is_seeded_inclusive_and_redraws_zero(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "small-range.yml"
    config_path.write_text(
        _config_text_with(a_min=-1, a_max=1, b_min=-1, b_max=1),
        encoding="utf-8",
    )
    config = load_config(config_path)

    first = list(generate_candidate_pairs(config, 20))
    second = list(generate_candidate_pairs(config, 20))

    assert first == second
    assert len(first) == 20
    assert {a for a, _ in first} == {-1, 1}
    assert {b for _, b in first} == {-1, 1}
    assert all(a != 0 and b != 0 for a, b in first)


def test_reference_quota_redistributes_unfilled_capacity_in_case_order() -> None:
    available = {
        "A1": 1,
        "A2": 10,
        "A3": 0,
        "A4": 10,
        "A5": 0,
        "A6": 2,
    }

    quota = allocate_reference_quota(available, reference_total=12)

    assert list(quota) == list(CASE_ORDER)
    assert quota == {
        "A1": 1,
        "A2": 5,
        "A3": 0,
        "A4": 4,
        "A5": 0,
        "A6": 2,
    }
    assert sum(quota.values()) == 12


def test_reference_quota_with_per_case_limit_stays_stratified_under_total_cap() -> None:
    available = {case: 1000 for case in CASE_ORDER}

    quota = allocate_reference_quota(
        available,
        reference_total=2000,
        limit_per_case=500,
    )

    assert quota == {
        "A1": 334,
        "A2": 334,
        "A3": 333,
        "A4": 333,
        "A5": 333,
        "A6": 333,
    }
    assert sum(quota.values()) == 2000
    assert max(quota.values()) <= 500


def test_stage_a_cli_processes_100_pairs_through_real_cpp_classifier(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "stage-a"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "run_stage_a.py"),
            "--config",
            str(CONFIG_PATH),
            "--limit",
            "100",
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stderr == ""
    with (output_dir / "stage_a_rows.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metadata = json.loads((output_dir / "stage_a_metadata.json").read_text(encoding="utf-8"))

    assert metadata["input_pairs"] == 100
    assert metadata["nonsingular_rows"] + metadata["singular_excluded"] == 100
    assert metadata["case_counts"] == {
        case: sum(row["case"] == case for row in rows)
        for case in CASE_ORDER
    }
    assert list(rows[0]) == [
        "sample_id",
        "a",
        "b",
        "discriminant",
        "c2",
        "has3",
        "case",
        "three_torsion_x",
    ]
    assert [row["sample_id"] for row in rows[:3]] == [
        "sample_000000001",
        "sample_000000002",
        "sample_000000003",
    ]
    assert all(row["case"] in CASE_ORDER for row in rows)


def test_stage_a_cli_excludes_one_allowed_zero_zero_singular_pair(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "zero.yml"
    config_path.write_text(
        _config_text_with(
            a_min=0,
            a_max=0,
            b_min=0,
            b_max=0,
            include_zero_coefficients=True,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "stage-a-zero"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "run_stage_a.py"),
            "--config",
            str(config_path),
            "--limit",
            "1",
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stderr == ""
    with (output_dir / "stage_a_rows.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames
    metadata = json.loads((output_dir / "stage_a_metadata.json").read_text(encoding="utf-8"))

    assert columns == [
        "sample_id",
        "a",
        "b",
        "discriminant",
        "c2",
        "has3",
        "case",
        "three_torsion_x",
    ]
    assert rows == []
    assert metadata["input_pairs"] == 1
    assert metadata["nonsingular_rows"] == 0
    assert metadata["singular_excluded"] == 1
    assert metadata["case_counts"] == {case: 0 for case in CASE_ORDER}


def test_sage_reference_cli_computes_exact_qq_torsion_for_all_cases(
    tmp_path: Path,
) -> None:
    sage = shutil.which("sage")
    if sage is None:
        pytest.skip("SageMath is not available")
    input_path = tmp_path / "stage_a_rows.csv"
    input_path.write_text(
        "sample_id,a,b,case\n"
        "sample_a1,0,-27648,A1\n"
        "sample_a2,-1947,108214,A2\n"
        "sample_a3,-24003,1296702,A3\n"
        "sample_a4,-432,8208,A4\n"
        "sample_a5,-44091,3304854,A5\n"
        "sample_a6,-9,0,A6\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "sage-reference"

    completed = subprocess.run(
        [
            sage,
            "-python",
            str(REPOSITORY_ROOT / "scripts" / "run_sage_reference.py"),
            "--config",
            str(CONFIG_PATH),
            "--limit-per-case",
            "1",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stderr == ""
    with (output_dir / "stage_b_reference_rows.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    selection = json.loads(
        (output_dir / "stage_b_selection.json").read_text(encoding="utf-8")
    )

    assert list(rows[0]) == [
        "sample_id",
        "a",
        "b",
        "case",
        "torsion_order",
        "torsion_invariants",
        "generators",
    ]
    assert [row["sample_id"] for row in rows] == [f"sample_a{i}" for i in range(1, 7)]
    assert [row["torsion_order"] for row in rows] == ["3", "12", "12", "5", "8", "4"]
    assert [row["torsion_invariants"] for row in rows] == [
        "[3]",
        "[12]",
        "[2,6]",
        "[5]",
        "[8]",
        "[2,2]",
    ]
    assert [row["generators"] for row in rows] == [
        '[["48","288","1"]]',
        '[["-37","360","1"]]',
        '[["39","648","1"],["66","0","1"]]',
        '[["-12","108","1"]]',
        '[["75","648","1"]]',
        '[["-3","0","1"],["0","0","1"]]',
    ]
    assert selection == {
        "available_per_case": {case: 1 for case in CASE_ORDER},
        "limit_per_case": 1,
        "reference_total": 2000,
        "seed": 20260220,
        "selected_per_case": {case: 1 for case in CASE_ORDER},
        "selected_total": 6,
    }
