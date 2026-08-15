from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "conference.yml"
CSV_COLUMNS = [
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
]
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
FORBIDDEN_ENVIRONMENT_KEYS = {
    "hostname",
    "username",
    "cwd",
    "absolute_paths",
    "git_dirty",
    "path",
    "timestamp",
    "environment",
}


def _run_benchmark(output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "run_benchmark.py"),
            "--config",
            str(CONFIG_PATH),
            "--sizes",
            "5",
            "10",
            "--sample-per-size",
            "4",
            "--repeats",
            "1",
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _load_results(output_dir: Path) -> tuple[list[str], list[dict[str, str]], dict[str, object]]:
    with (output_dir / "benchmark_scaling.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames
    environment = json.loads(
        (output_dir / "environment.json").read_text(encoding="utf-8")
    )
    assert columns is not None
    return columns, rows, environment


def test_benchmark_cli_uses_real_backends_and_writes_stable_publication_data(
    tmp_path: Path,
) -> None:
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first = _run_benchmark(first_output)
    second = _run_benchmark(second_output)

    assert first.stderr == ""
    assert second.stderr == ""
    first_columns, first_rows, first_environment = _load_results(first_output)
    second_columns, second_rows, second_environment = _load_results(second_output)

    assert first_columns == CSV_COLUMNS
    assert second_columns == CSV_COLUMNS
    assert [row["range_limit"] for row in first_rows] == ["5", "10"]
    assert [row["n"] for row in first_rows] == ["4", "4"]
    assert [row["repeats"] for row in first_rows] == ["1", "1"]
    assert [row["seed"] for row in first_rows] == [
        "11414379241583312001",
        "14814945309683905240",
    ]
    assert [row["input_sha256"] for row in first_rows] == [
        row["input_sha256"] for row in second_rows
    ]
    assert all(len(row["input_sha256"]) == 64 for row in first_rows)

    timing_columns = [
        f"{backend}_{clock}_median_s"
        for backend in ("python", "cpp", "sage")
        for clock in ("wall", "cpu")
    ]
    for row in first_rows:
        assert all(float(row[column]) > 0 for column in timing_columns)
        assert int(row["python_cpp_mismatches"]) == 0
        assert int(row["python_sage_mismatches"]) == 0
        assert float(row["python_over_cpp_wall_x"]) == pytest.approx(
            float(row["python_wall_median_s"]) / float(row["cpp_wall_median_s"]),
            rel=1e-9,
        )
        assert float(row["python_over_cpp_cpu_x"]) == pytest.approx(
            float(row["python_cpu_median_s"]) / float(row["cpp_cpu_median_s"]),
            rel=1e-9,
        )
        assert float(row["sage_over_cpp_wall_x"]) == pytest.approx(
            float(row["sage_wall_median_s"]) / float(row["cpp_wall_median_s"]),
            rel=1e-9,
        )
        assert float(row["sage_over_cpp_cpu_x"]) == pytest.approx(
            float(row["sage_cpu_median_s"]) / float(row["cpp_cpu_median_s"]),
            rel=1e-9,
        )
        assert float(row["sage_over_python_wall_x"]) == pytest.approx(
            float(row["sage_wall_median_s"]) / float(row["python_wall_median_s"]),
            rel=1e-9,
        )
        assert float(row["sage_over_python_cpu_x"]) == pytest.approx(
            float(row["sage_cpu_median_s"]) / float(row["python_cpu_median_s"]),
            rel=1e-9,
        )
        assert all(
            float(row[column]) >= 0
            for column in (
                "two_torsion_candidates_mean",
                "three_torsion_candidates_mean",
                "exact_polynomial_checks_mean",
                "square_checks_mean",
            )
        )

    assert set(first_environment) == ENVIRONMENT_KEYS
    assert set(second_environment) == ENVIRONMENT_KEYS
    assert set(first_environment).isdisjoint(FORBIDDEN_ENVIRONMENT_KEYS)
    assert set(second_environment).isdisjoint(FORBIDDEN_ENVIRONMENT_KEYS)
    assert first_environment == second_environment
    assert first_environment["sage_version"] == "10.8"
    assert int(first_environment["logical_cores"]) > 0
    assert float(first_environment["memory_gib"]) > 0


def test_benchmark_rejects_more_than_exact_nonsingular_capacity_before_cmake(
    tmp_path: Path,
) -> None:
    test_bin = tmp_path / "bin"
    test_bin.mkdir()
    cmake_marker = tmp_path / "cmake-called"
    cmake_sentinel = test_bin / "cmake"
    cmake_sentinel.write_text(
        "#!/bin/sh\n: > \"$CMAKE_SENTINEL_MARKER\"\nexit 99\n",
        encoding="utf-8",
    )
    cmake_sentinel.chmod(0o755)
    environment = os.environ.copy()
    environment["CMAKE_SENTINEL_MARKER"] = str(cmake_marker)
    environment["PATH"] = os.pathsep.join((str(test_bin), environment["PATH"]))
    assert shutil.which("cmake", path=environment["PATH"]) == str(cmake_sentinel)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "run_benchmark.py"),
            "--config",
            str(CONFIG_PATH),
            "--sizes",
            "3",
            "--sample-per-size",
            "35",
            "--repeats",
            "1",
            "--output-dir",
            str(tmp_path / "output"),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=environment,
    )

    assert completed.returncode != 0
    assert "exact nonsingular nonzero capacity 34 for range limit 3" in completed.stderr
    assert not (tmp_path / "output").exists()
    assert not cmake_marker.exists()


def test_cpp_benchmark_build_configures_real_release_cache(tmp_path: Path) -> None:
    sage = shutil.which("sage")
    assert sage is not None
    build_dir = tmp_path / "release-build"
    inspection = (
        "import runpy, sys; from pathlib import Path; "
        "namespace = runpy.run_path(sys.argv[1], run_name='benchmark_test'); "
        "namespace['_build_classifier'](Path(sys.argv[2]))"
    )

    subprocess.run(
        [
            sage,
            "-python",
            "-c",
            inspection,
            str(REPOSITORY_ROOT / "scripts" / "run_benchmark.py"),
            str(build_dir),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    cache_entries = (build_dir / "CMakeCache.txt").read_text(encoding="utf-8")
    assert "CMAKE_BUILD_TYPE:STRING=Release" in cache_entries.splitlines()
