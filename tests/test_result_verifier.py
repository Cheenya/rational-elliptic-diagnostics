from __future__ import annotations

import csv
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "conference.yml"
VERIFY_SCRIPT = REPOSITORY_ROOT / "scripts" / "verify_results.py"

STAGE_A_SUMMARY_COLUMNS = [
    "case",
    "c2",
    "has3",
    "count",
    "share_nonsingular",
    "input_pairs",
    "nonsingular_rows",
    "singular_excluded",
    "seed",
]
STAGE_B_COLUMNS = [
    "sample_id",
    "a",
    "b",
    "case",
    "torsion_order",
    "torsion_invariants",
    "generators",
]
CALIBRATION_COLUMNS = [
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
]
FORBIDDEN_FIELDS = {
    "hostname",
    "username",
    "cwd",
    "absolute_paths",
    "git_dirty",
    "timestamp",
}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames is not None
        return reader.fieldnames, rows


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def valid_runtime_inputs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sage = shutil.which("sage")
    assert sage is not None
    output_dir = tmp_path_factory.mktemp("result-verifier-input")
    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "run_stage_a.py"),
            "--config",
            str(CONFIG_PATH),
            "--limit",
            "40",
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sage,
            "-python",
            str(REPOSITORY_ROOT / "scripts" / "run_sage_reference.py"),
            "--config",
            str(CONFIG_PATH),
            "--limit-per-case",
            "2",
            "--input",
            str(output_dir / "stage_a_rows.csv"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return output_dir


def _copy_runtime_inputs(source: Path, destination: Path) -> None:
    destination.mkdir()
    for filename in (
        "stage_a_rows.csv",
        "stage_a_metadata.json",
        "stage_b_reference_rows.csv",
        "stage_b_selection.json",
    ):
        shutil.copy2(source / filename, destination / filename)


def _prepare_results(destination: Path) -> None:
    destination.mkdir()
    for filename in ("benchmark_scaling.csv", "environment.json"):
        shutil.copy2(REPOSITORY_ROOT / "results" / filename, destination / filename)


def _run_verifier(input_dir: Path, results_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            "--config",
            str(CONFIG_PATH),
            "--input-dir",
            str(input_dir),
            "--results-dir",
            str(results_dir),
            "--calibration-per-class",
            "2",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_verifier_recomputes_reference_and_writes_stable_publication_files(
    tmp_path: Path,
    valid_runtime_inputs: Path,
) -> None:
    input_dir = tmp_path / "input"
    results_dir = tmp_path / "results"
    _copy_runtime_inputs(valid_runtime_inputs, input_dir)
    _prepare_results(results_dir)

    completed = _run_verifier(input_dir, results_dir)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines()[:4] == [
        "Python/C++ mismatches: 0",
        "Singular curves excluded: true",
        "Sage reference rows valid: true",
        "Required summaries present: true",
    ]
    assert completed.stderr == ""
    assert sorted(path.name for path in results_dir.iterdir()) == [
        "benchmark_scaling.csv",
        "calibration.csv",
        "environment.json",
        "stage_a_summary.csv",
        "stage_b_reference.csv",
    ]

    summary_columns, summary_rows = _read_csv(results_dir / "stage_a_summary.csv")
    reference_columns, reference_rows = _read_csv(results_dir / "stage_b_reference.csv")
    calibration_columns, calibration_rows = _read_csv(results_dir / "calibration.csv")

    assert summary_columns == STAGE_A_SUMMARY_COLUMNS
    assert [row["case"] for row in summary_rows] == [
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
    ]
    assert all(len(row["share_nonsingular"].partition(".")[2]) == 12 for row in summary_rows)
    stage_a_metadata = json.loads(
        (input_dir / "stage_a_metadata.json").read_text(encoding="utf-8")
    )
    assert sum(int(row["count"]) for row in summary_rows) == stage_a_metadata["nonsingular_rows"]
    assert (
        stage_a_metadata["nonsingular_rows"] + stage_a_metadata["singular_excluded"]
        == stage_a_metadata["input_pairs"]
    )
    assert reference_columns == STAGE_B_COLUMNS
    assert reference_rows == sorted(
        reference_rows,
        key=lambda row: (int(row["case"][1:]), row["sample_id"]),
    )
    stage_b_selection = json.loads(
        (input_dir / "stage_b_selection.json").read_text(encoding="utf-8")
    )
    assert len(reference_rows) == stage_b_selection["selected_total"]
    assert len(reference_rows) <= stage_b_selection["reference_total"]
    assert {
        case: sum(row["case"] == case for row in reference_rows)
        for case in ("A1", "A2", "A3", "A4", "A5", "A6")
    } == stage_b_selection["selected_per_case"]
    assert calibration_columns == CALIBRATION_COLUMNS
    assert [row["backend"] for row in calibration_rows] == ["python", "cpp"]
    assert all(row["n"] == "4" for row in calibration_rows)
    assert all(row["truth_positive"] == "2" for row in calibration_rows)
    assert all(row["truth_negative"] == "2" for row in calibration_rows)
    assert all(row["python_cpp_mismatches"] == "0" for row in calibration_rows)
    assert calibration_rows[0]["input_sha256"] == calibration_rows[1]["input_sha256"]
    assert len(calibration_rows[0]["input_sha256"]) == 64
    for columns in (summary_columns, reference_columns, calibration_columns):
        assert set(columns).isdisjoint(FORBIDDEN_FIELDS)
    publication_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(results_dir.iterdir())
    )
    assert str(REPOSITORY_ROOT) not in publication_text
    assert Path.home().name not in publication_text


def test_verifier_rejects_singular_stage_a_row(
    tmp_path: Path,
    valid_runtime_inputs: Path,
) -> None:
    input_dir = tmp_path / "input"
    results_dir = tmp_path / "results"
    _copy_runtime_inputs(valid_runtime_inputs, input_dir)
    _prepare_results(results_dir)
    columns, rows = _read_csv(input_dir / "stage_a_rows.csv")
    rows[0].update(
        {
            "a": "0",
            "b": "0",
            "discriminant": "0",
            "c2": "0",
            "has3": "0",
            "case": "A4",
            "three_torsion_x": "-",
        }
    )
    with (input_dir / "stage_a_rows.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    completed = _run_verifier(input_dir, results_dir)

    assert completed.returncode != 0
    assert "singular Stage A row" in completed.stderr


def test_verifier_rejects_corrupted_sage_invariants(
    tmp_path: Path,
    valid_runtime_inputs: Path,
) -> None:
    input_dir = tmp_path / "input"
    results_dir = tmp_path / "results"
    _copy_runtime_inputs(valid_runtime_inputs, input_dir)
    _prepare_results(results_dir)
    columns, rows = _read_csv(input_dir / "stage_b_reference_rows.csv")
    assert rows
    rows[0]["torsion_invariants"] = "[999]"
    with (input_dir / "stage_b_reference_rows.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    completed = _run_verifier(input_dir, results_dir)

    assert completed.returncode != 0
    assert "Sage reference mismatch" in completed.stderr


def test_verifier_rejects_joint_stage_a_case_and_c2_substitution(
    tmp_path: Path,
    valid_runtime_inputs: Path,
) -> None:
    input_dir = tmp_path / "input"
    results_dir = tmp_path / "results"
    _copy_runtime_inputs(valid_runtime_inputs, input_dir)
    _prepare_results(results_dir)
    columns, rows = _read_csv(input_dir / "stage_a_rows.csv")
    changed = next(row for row in rows if row["case"] == "A4")
    changed["c2"] = "1"
    changed["case"] = "A5"
    _write_csv(input_dir / "stage_a_rows.csv", columns, rows)
    metadata = json.loads((input_dir / "stage_a_metadata.json").read_text(encoding="utf-8"))
    metadata["case_counts"]["A4"] -= 1
    metadata["case_counts"]["A5"] += 1
    _write_json(input_dir / "stage_a_metadata.json", metadata)

    completed = _run_verifier(input_dir, results_dir)

    assert completed.returncode != 0
    assert "Stage A classifier mismatch" in completed.stderr


def test_verifier_rejects_hidden_three_torsion_root_on_valid_a1_curve(
    tmp_path: Path,
    valid_runtime_inputs: Path,
) -> None:
    input_dir = tmp_path / "input"
    results_dir = tmp_path / "results"
    _copy_runtime_inputs(valid_runtime_inputs, input_dir)
    _prepare_results(results_dir)
    columns, rows = _read_csv(input_dir / "stage_a_rows.csv")
    rows[0].update(
        {
            "a": "-21",
            "b": "37",
            "discriminant": "1296",
            "c2": "0",
            "has3": "0",
            "case": "A4",
            "three_torsion_x": "-",
        }
    )
    _write_csv(input_dir / "stage_a_rows.csv", columns, rows)

    completed = _run_verifier(input_dir, results_dir)

    assert completed.returncode != 0
    assert "Stage A classifier mismatch" in completed.stderr


@pytest.mark.parametrize(
    ("a", "message"),
    [
        (10001, "Stage A coefficient outside configured bounds"),
        (0, "Stage A zero coefficient violates configuration"),
    ],
)
def test_verifier_rejects_stage_a_bounds_and_zero_policy_violations(
    tmp_path: Path,
    valid_runtime_inputs: Path,
    a: int,
    message: str,
) -> None:
    input_dir = tmp_path / "input"
    results_dir = tmp_path / "results"
    _copy_runtime_inputs(valid_runtime_inputs, input_dir)
    _prepare_results(results_dir)
    columns, rows = _read_csv(input_dir / "stage_a_rows.csv")
    b = int(rows[0]["b"])
    rows[0]["a"] = str(a)
    rows[0]["discriminant"] = str(-16 * (4 * a**3 + 27 * b**2))
    _write_csv(input_dir / "stage_a_rows.csv", columns, rows)

    completed = _run_verifier(input_dir, results_dir)

    assert completed.returncode != 0
    assert message in completed.stderr


def _recompute_single_sage_row(
    tmp_path: Path,
    row: dict[str, str],
) -> dict[str, str]:
    sage = shutil.which("sage")
    assert sage is not None
    input_path = tmp_path / "single-stage-a.csv"
    input_path.write_text(
        "sample_id,a,b,case\n"
        f"{row['sample_id']},{row['a']},{row['b']},{row['case']}\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "single-sage-output"
    subprocess.run(
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
    _, computed = _read_csv(output_dir / "stage_b_reference_rows.csv")
    assert len(computed) == 1
    return computed[0]


def test_verifier_rejects_valid_but_nonseeded_stage_b_substitution(
    tmp_path: Path,
    valid_runtime_inputs: Path,
) -> None:
    input_dir = tmp_path / "input"
    results_dir = tmp_path / "results"
    _copy_runtime_inputs(valid_runtime_inputs, input_dir)
    _prepare_results(results_dir)
    stage_a_columns, stage_a_rows = _read_csv(input_dir / "stage_a_rows.csv")
    del stage_a_columns
    stage_b_columns, stage_b_rows = _read_csv(input_dir / "stage_b_reference_rows.csv")
    selected_ids = {row["sample_id"] for row in stage_b_rows}
    replaced = stage_b_rows[0]
    replacement_source = next(
        row
        for row in stage_a_rows
        if row["case"] == replaced["case"] and row["sample_id"] not in selected_ids
    )
    replacement = _recompute_single_sage_row(tmp_path, replacement_source)
    stage_b_rows[0] = replacement
    stage_b_rows.sort(key=lambda row: (int(row["case"][1:]), row["sample_id"]))
    _write_csv(input_dir / "stage_b_reference_rows.csv", stage_b_columns, stage_b_rows)

    completed = _run_verifier(input_dir, results_dir)

    assert completed.returncode != 0
    assert "Stage B seeded selection mismatch" in completed.stderr


def test_verifier_rejects_corrupted_benchmark_schema(
    tmp_path: Path,
    valid_runtime_inputs: Path,
) -> None:
    input_dir = tmp_path / "input"
    results_dir = tmp_path / "results"
    _copy_runtime_inputs(valid_runtime_inputs, input_dir)
    _prepare_results(results_dir)
    benchmark_path = results_dir / "benchmark_scaling.csv"
    lines = benchmark_path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("range_limit", "range", 1)
    benchmark_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    completed = _run_verifier(input_dir, results_dir)

    assert completed.returncode != 0
    assert "invalid schema for benchmark_scaling.csv" in completed.stderr


def test_verifier_rejects_path_like_environment_value(
    tmp_path: Path,
    valid_runtime_inputs: Path,
) -> None:
    input_dir = tmp_path / "input"
    results_dir = tmp_path / "results"
    _copy_runtime_inputs(valid_runtime_inputs, input_dir)
    _prepare_results(results_dir)
    environment_path = results_dir / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["cpu_model"] = str(Path("/").joinpath("private", "example"))
    _write_json(environment_path, environment)

    completed = _run_verifier(input_dir, results_dir)

    assert completed.returncode != 0
    assert "path-like value in environment.json" in completed.stderr


def test_verifier_rejects_short_stage_a_row_as_invalid_schema(
    tmp_path: Path,
    valid_runtime_inputs: Path,
) -> None:
    input_dir = tmp_path / "input"
    results_dir = tmp_path / "results"
    _copy_runtime_inputs(valid_runtime_inputs, input_dir)
    _prepare_results(results_dir)
    stage_a_path = input_dir / "stage_a_rows.csv"
    lines = stage_a_path.read_text(encoding="utf-8").splitlines()
    lines[1] = ",".join(lines[1].split(",")[:4])
    stage_a_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    completed = _run_verifier(input_dir, results_dir)

    assert completed.returncode != 0
    assert "invalid row schema for stage_a_rows.csv" in completed.stderr


@pytest.mark.parametrize(
    "filename",
    ["stage_a_rows.csv", "benchmark_scaling.csv"],
)
def test_verifier_rejects_overlong_csv_data_row(
    tmp_path: Path,
    valid_runtime_inputs: Path,
    filename: str,
) -> None:
    input_dir = tmp_path / "input"
    results_dir = tmp_path / "results"
    _copy_runtime_inputs(valid_runtime_inputs, input_dir)
    _prepare_results(results_dir)
    path = input_dir / filename if filename.startswith("stage_a") else results_dir / filename
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] += ",unexpected"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    completed = _run_verifier(input_dir, results_dir)

    assert completed.returncode != 0
    assert f"invalid row schema for {filename}" in completed.stderr


def test_verifier_rejects_integral_float_stage_a_seed(
    tmp_path: Path,
    valid_runtime_inputs: Path,
) -> None:
    input_dir = tmp_path / "input"
    results_dir = tmp_path / "results"
    _copy_runtime_inputs(valid_runtime_inputs, input_dir)
    _prepare_results(results_dir)
    metadata_path = input_dir / "stage_a_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["seed"] = float(metadata["seed"])
    _write_json(metadata_path, metadata)

    completed = _run_verifier(input_dir, results_dir)

    assert completed.returncode != 0
    assert "stage_a_metadata.seed must be an integer" in completed.stderr


@pytest.mark.parametrize("field", ["seed", "reference_total"])
def test_verifier_rejects_integral_float_stage_b_integer_contracts(
    tmp_path: Path,
    valid_runtime_inputs: Path,
    field: str,
) -> None:
    input_dir = tmp_path / "input"
    results_dir = tmp_path / "results"
    _copy_runtime_inputs(valid_runtime_inputs, input_dir)
    _prepare_results(results_dir)
    selection_path = input_dir / "stage_b_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection[field] = float(selection[field])
    _write_json(selection_path, selection)

    completed = _run_verifier(input_dir, results_dir)

    assert completed.returncode != 0
    assert f"stage_b_selection.{field} must be an integer" in completed.stderr


@pytest.mark.parametrize(
    ("filename", "field"),
    [
        ("stage_a_metadata.json", "seed"),
        ("stage_b_selection.json", "seed"),
        ("stage_b_selection.json", "reference_total"),
    ],
)
def test_verifier_rejects_boolean_metadata_integer_contracts(
    tmp_path: Path,
    valid_runtime_inputs: Path,
    filename: str,
    field: str,
) -> None:
    input_dir = tmp_path / "input"
    results_dir = tmp_path / "results"
    _copy_runtime_inputs(valid_runtime_inputs, input_dir)
    _prepare_results(results_dir)
    path = input_dir / filename
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata[field] = True
    _write_json(path, metadata)

    completed = _run_verifier(input_dir, results_dir)

    prefix = "stage_a_metadata" if filename.startswith("stage_a") else "stage_b_selection"
    assert completed.returncode != 0
    assert f"{prefix}.{field} must be an integer" in completed.stderr


def test_result_set_publication_rolls_back_after_commit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = runpy.run_path(str(VERIFY_SCRIPT), run_name="result_verifier_test")
    publish_result_set = namespace["_publish_result_set"]
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    old_files = {
        filename: f"old:{filename}\n".encode("utf-8")
        for filename in (
            "benchmark_scaling.csv",
            "calibration.csv",
            "environment.json",
            "stage_a_summary.csv",
            "stage_b_reference.csv",
        )
    }
    for filename, content in old_files.items():
        (results_dir / filename).write_bytes(content)
    new_files = {
        filename: f"new:{filename}\n"
        for filename in old_files
    }
    real_replace = Path.replace
    failed = False

    def fail_one_staged_commit(source: Path, destination: Path) -> Path:
        nonlocal failed
        destination = Path(destination)
        if (
            not failed
            and source.parent.name.startswith(".results.stage-")
            and destination.parent == results_dir
            and source.name == "stage_b_reference.csv"
        ):
            failed = True
            raise OSError("injected publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_one_staged_commit)

    with pytest.raises(OSError, match="injected publication failure"):
        publish_result_set(results_dir, new_files)

    assert {
        path.name: path.read_bytes()
        for path in results_dir.iterdir()
    } == old_files
    assert not [
        path
        for path in tmp_path.iterdir()
        if path.name.startswith((".results.stage-", ".results.backup-"))
    ]
