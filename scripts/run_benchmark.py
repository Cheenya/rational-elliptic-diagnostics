from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from io import StringIO
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time


try:
    from sage.all import EllipticCurve, QQ
    from sage.env import SAGE_VERSION
except ModuleNotFoundError:
    sage_executable = shutil.which("sage")
    if sage_executable is None:
        raise RuntimeError("SageMath is required and 'sage' is not on PATH") from None
    os.execv(
        sage_executable,
        [sage_executable, "-python", str(Path(__file__).resolve()), *sys.argv[1:]],
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from elliptic_diagnostics.classifier import (
    _exact_integer_cube_root,
    _integer_root_candidates_for_constant,
    classify_a1_a6,
)
from elliptic_diagnostics.curve import RationalCurve
from elliptic_diagnostics.experiment import ExperimentConfig, load_config


CSV_COLUMNS = (
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
ENVIRONMENT_KEYS = (
    "os",
    "architecture",
    "cpu_model",
    "logical_cores",
    "memory_gib",
    "sage_version",
    "python_version",
    "compiler_version",
)


def _root_relative(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _seed_for_limit(seed: int, limit: int) -> int:
    digest = sha256(f"{seed}:{limit}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _floor_cube_root(value: int) -> int:
    low = 0
    high = 1
    while high**3 <= value:
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        if middle**3 <= value:
            low = middle
        else:
            high = middle
    return low


def _valid_pair_capacity(limit: int) -> int:
    singular_parameter_limit = min(
        math.isqrt(limit // 3),
        _floor_cube_root(limit // 2),
    )
    return (2 * limit) ** 2 - 2 * singular_parameter_limit


def _generate_pairs(seed: int, limit: int, count: int) -> list[tuple[int, int]]:
    capacity = _valid_pair_capacity(limit)
    if count > capacity:
        raise ValueError(
            f"cannot draw {count} pairs from exact nonsingular nonzero capacity "
            f"{capacity} for range limit {limit}"
        )
    random_source = random.Random(_seed_for_limit(seed, limit))
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(pairs) < count:
        pair = (
            random_source.randint(-limit, limit),
            random_source.randint(-limit, limit),
        )
        if pair in seen or pair[0] == 0 or pair[1] == 0:
            continue
        if not RationalCurve(*pair).is_nonsingular():
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def _input_sha256(pairs: list[tuple[int, int]]) -> str:
    payload = "".join(f"{a}\t{b}\n" for a, b in pairs).encode("ascii")
    return sha256(payload).hexdigest()


def _build_classifier(build_dir: Path) -> Path:
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


def _classifier_input(pairs: list[tuple[int, int]]) -> str:
    return "".join(
        f"sample_{index:09d}\t{a}\t{b}\n"
        for index, (a, b) in enumerate(pairs, start=1)
    )


def _parse_cpp_results(
    output: str,
    expected_count: int,
) -> list[tuple[int, bool, str]]:
    lines = output.splitlines()
    if len(lines) != expected_count:
        raise RuntimeError("C++ classifier returned an invalid row count")
    results: list[tuple[int, bool, str]] = []
    for index, line in enumerate(lines, start=1):
        fields = line.split("\t")
        if len(fields) != 5 or fields[0] != f"sample_{index:09d}":
            raise RuntimeError("C++ classifier returned an invalid schema or row order")
        c2_text, has3_text, case = fields[1:4]
        if c2_text not in {"0", "1", "3"} or has3_text not in {"0", "1"}:
            raise RuntimeError("C++ classifier returned invalid diagnostic values")
        results.append((int(c2_text), has3_text == "1", case))
    return results


def _time_python(
    pairs: list[tuple[int, int]],
) -> tuple[float, float, list[tuple[int, bool, str]]]:
    cpu_start = time.process_time_ns()
    wall_start = time.perf_counter_ns()
    results = []
    for a, b in pairs:
        result = classify_a1_a6(RationalCurve(a, b))
        results.append((result.c2, result.has_3_torsion_indicator, result.case))
    wall_seconds = (time.perf_counter_ns() - wall_start) / 1_000_000_000
    cpu_seconds = (time.process_time_ns() - cpu_start) / 1_000_000_000
    return wall_seconds, cpu_seconds, results


def _children_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def _time_cpp(
    executable: Path,
    pairs: list[tuple[int, int]],
) -> tuple[float, float, list[tuple[int, bool, str]]]:
    classifier_input = _classifier_input(pairs)
    cpu_start = _children_cpu_seconds()
    wall_start = time.perf_counter_ns()
    completed = subprocess.run(
        [str(executable)],
        input=classifier_input,
        check=True,
        capture_output=True,
        text=True,
    )
    wall_seconds = (time.perf_counter_ns() - wall_start) / 1_000_000_000
    cpu_seconds = _children_cpu_seconds() - cpu_start
    return (
        wall_seconds,
        cpu_seconds,
        _parse_cpp_results(completed.stdout, len(pairs)),
    )


def _case_from_exact_invariants(c2: int, has3: bool) -> str:
    if has3 and c2 == 0:
        return "A1"
    if has3 and c2 == 1:
        return "A2"
    if has3 and c2 == 3:
        return "A3"
    if not has3 and c2 == 0:
        return "A4"
    if not has3 and c2 == 1:
        return "A5"
    if not has3 and c2 == 3:
        return "A6"
    raise RuntimeError(f"unexpected exact torsion diagnostics: c2={c2}, has3={has3}")


def _time_sage(
    pairs: list[tuple[int, int]],
) -> tuple[float, float, list[tuple[int, bool, str]]]:
    cpu_start = time.process_time_ns()
    wall_start = time.perf_counter_ns()
    results: list[tuple[int, bool, str]] = []
    for a, b in pairs:
        torsion = EllipticCurve(QQ, [0, 0, 0, a, b]).torsion_subgroup()
        invariants = [int(value) for value in torsion.invariants()]
        c2 = math.prod(math.gcd(value, 2) for value in invariants) - 1
        has3 = any(value % 3 == 0 for value in invariants)
        results.append((c2, has3, _case_from_exact_invariants(c2, has3)))
    wall_seconds = (time.perf_counter_ns() - wall_start) / 1_000_000_000
    cpu_seconds = (time.process_time_ns() - cpu_start) / 1_000_000_000
    return wall_seconds, cpu_seconds, results


def _candidate_counts(curve: RationalCurve) -> tuple[int, int, int, int]:
    two_candidates = set(_integer_root_candidates_for_constant(curve.b))
    if curve.b == 0 and -curve.a >= 0:
        square_root = math.isqrt(-curve.a)
        if square_root * square_root == -curve.a:
            two_candidates.update((-square_root, square_root))

    if curve.a != 0:
        three_candidates = _integer_root_candidates_for_constant(-(curve.a**2))
    else:
        three_candidates = [0]
        cubic_root = _exact_integer_cube_root(-4 * curve.b)
        if cubic_root is not None:
            three_candidates.append(cubic_root)

    square_checks = sum(
        curve.third_division_polynomial(candidate) == 0
        for candidate in three_candidates
    )
    return (
        len(two_candidates),
        len(three_candidates),
        len(two_candidates) + len(three_candidates),
        square_checks,
    )


def _workload_means(pairs: list[tuple[int, int]]) -> tuple[float, float, float, float]:
    totals = [0, 0, 0, 0]
    for a, b in pairs:
        for index, value in enumerate(_candidate_counts(RationalCurve(a, b))):
            totals[index] += value
    return tuple(total / len(pairs) for total in totals)  # type: ignore[return-value]


def _mismatch_count(
    expected: list[tuple[int, bool, str]],
    actual: list[tuple[int, bool, str]],
) -> int:
    if len(expected) != len(actual):
        raise RuntimeError("backend result lengths differ")
    return sum(left != right for left, right in zip(expected, actual, strict=True))


def _median_timings(
    executable: Path,
    pairs: list[tuple[int, int]],
    repeats: int,
) -> tuple[dict[str, float], int, int]:
    python_runs = [_time_python(pairs) for _ in range(repeats)]
    cpp_runs = [_time_cpp(executable, pairs) for _ in range(repeats)]
    sage_runs = [_time_sage(pairs) for _ in range(repeats)]

    expected = python_runs[0][2]
    if any(run[2] != expected for run in python_runs[1:]):
        raise RuntimeError("Python results changed between benchmark repetitions")
    if any(run[2] != cpp_runs[0][2] for run in cpp_runs[1:]):
        raise RuntimeError("C++ results changed between benchmark repetitions")
    if any(run[2] != sage_runs[0][2] for run in sage_runs[1:]):
        raise RuntimeError("Sage results changed between benchmark repetitions")

    timings = {
        "python_wall": statistics.median(run[0] for run in python_runs),
        "python_cpu": statistics.median(run[1] for run in python_runs),
        "cpp_wall": statistics.median(run[0] for run in cpp_runs),
        "cpp_cpu": statistics.median(run[1] for run in cpp_runs),
        "sage_wall": statistics.median(run[0] for run in sage_runs),
        "sage_cpu": statistics.median(run[1] for run in sage_runs),
    }
    if any(value <= 0 for value in timings.values()):
        raise RuntimeError("a backend produced a nonpositive measured duration")
    return (
        timings,
        _mismatch_count(expected, cpp_runs[0][2]),
        _mismatch_count(expected, sage_runs[0][2]),
    )


def _format_decimal(value: float) -> str:
    return f"{value:.12f}"


def _benchmark_row(
    executable: Path,
    config: ExperimentConfig,
    limit: int,
    sample_per_size: int,
    repeats: int,
) -> dict[str, str | int]:
    pairs = _generate_pairs(config.seed, limit, sample_per_size)

    classify_a1_a6(RationalCurve(*pairs[0]))
    _time_cpp(executable, pairs[:1])
    _time_sage(pairs[:1])

    timings, python_cpp_mismatches, python_sage_mismatches = _median_timings(
        executable,
        pairs,
        repeats,
    )
    workload = _workload_means(pairs)
    rounded = {key: float(_format_decimal(value)) for key, value in timings.items()}

    return {
        "range_limit": limit,
        "n": sample_per_size,
        "repeats": repeats,
        "seed": _seed_for_limit(config.seed, limit),
        "input_sha256": _input_sha256(pairs),
        "python_wall_median_s": _format_decimal(rounded["python_wall"]),
        "python_cpu_median_s": _format_decimal(rounded["python_cpu"]),
        "cpp_wall_median_s": _format_decimal(rounded["cpp_wall"]),
        "cpp_cpu_median_s": _format_decimal(rounded["cpp_cpu"]),
        "sage_wall_median_s": _format_decimal(rounded["sage_wall"]),
        "sage_cpu_median_s": _format_decimal(rounded["sage_cpu"]),
        "python_over_cpp_wall_x": _format_decimal(
            rounded["python_wall"] / rounded["cpp_wall"]
        ),
        "python_over_cpp_cpu_x": _format_decimal(
            rounded["python_cpu"] / rounded["cpp_cpu"]
        ),
        "sage_over_cpp_wall_x": _format_decimal(
            rounded["sage_wall"] / rounded["cpp_wall"]
        ),
        "sage_over_cpp_cpu_x": _format_decimal(
            rounded["sage_cpu"] / rounded["cpp_cpu"]
        ),
        "sage_over_python_wall_x": _format_decimal(
            rounded["sage_wall"] / rounded["python_wall"]
        ),
        "sage_over_python_cpu_x": _format_decimal(
            rounded["sage_cpu"] / rounded["python_cpu"]
        ),
        "two_torsion_candidates_mean": _format_decimal(workload[0]),
        "three_torsion_candidates_mean": _format_decimal(workload[1]),
        "exact_polynomial_checks_mean": _format_decimal(workload[2]),
        "square_checks_mean": _format_decimal(workload[3]),
        "python_cpp_mismatches": python_cpp_mismatches,
        "python_sage_mismatches": python_sage_mismatches,
    }


def _cpu_model() -> str:
    commands = (
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        ["sysctl", "-n", "hw.model"],
    )
    for command in commands:
        completed = subprocess.run(command, capture_output=True, text=True)
        value = completed.stdout.strip()
        if completed.returncode == 0 and value:
            return value.splitlines()[0].strip()
    processor = platform.processor().strip()
    return processor or "unknown"


def _memory_gib() -> float:
    completed = subprocess.run(
        ["sysctl", "-n", "hw.memsize"],
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0 and completed.stdout.strip().isdigit():
        return int(completed.stdout.strip()) / (1024**3)
    page_size = os.sysconf("SC_PAGE_SIZE")
    page_count = os.sysconf("SC_PHYS_PAGES")
    return page_size * page_count / (1024**3)


def _compiler_version() -> str:
    completed = subprocess.run(
        ["c++", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    first_line = completed.stdout.splitlines()[0].strip()
    return " ".join(first_line.split())


def _environment() -> dict[str, object]:
    environment: dict[str, object] = {
        "os": platform.system(),
        "architecture": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cores": os.cpu_count() or 1,
        "memory_gib": round(_memory_gib(), 6),
        "sage_version": str(SAGE_VERSION),
        "python_version": platform.python_version(),
        "compiler_version": _compiler_version(),
    }
    if tuple(environment) != ENVIRONMENT_KEYS:
        raise RuntimeError("environment schema is not stable")
    return environment


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
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


def _validate_arguments(
    parser: argparse.ArgumentParser,
    config: ExperimentConfig,
    sizes: list[int],
    sample_per_size: int,
    repeats: int,
) -> None:
    if sample_per_size <= 0:
        parser.error("--sample-per-size must be positive")
    if repeats <= 0:
        parser.error("--repeats must be positive")
    for limit in sizes:
        if limit <= 0:
            parser.error("--sizes values must be positive")
        if (
            -limit < config.calibration_a_min
            or limit > config.calibration_a_max
            or -limit < config.calibration_b_min
            or limit > config.calibration_b_max
        ):
            parser.error(f"range limit {limit} is outside configured calibration bounds")
        capacity = _valid_pair_capacity(limit)
        if sample_per_size > capacity:
            parser.error(
                f"--sample-per-size {sample_per_size} exceeds exact nonsingular "
                f"nonzero capacity {capacity} for range limit {limit}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--sample-per-size", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", default="results")
    arguments = parser.parse_args()

    config = load_config(_root_relative(arguments.config))
    _validate_arguments(
        parser,
        config,
        arguments.sizes,
        arguments.sample_per_size,
        arguments.repeats,
    )
    output_dir = _root_relative(arguments.output_dir)

    rows: list[dict[str, str | int]] = []
    with tempfile.TemporaryDirectory(prefix="elliptic-diagnostics-benchmark-") as directory:
        executable = _build_classifier(Path(directory))
        for limit in arguments.sizes:
            rows.append(
                _benchmark_row(
                    executable,
                    config,
                    limit,
                    arguments.sample_per_size,
                    arguments.repeats,
                )
            )

    csv_buffer = StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _write_text_atomic(output_dir / "benchmark_scaling.csv", csv_buffer.getvalue())
    _write_text_atomic(
        output_dir / "environment.json",
        json.dumps(_environment(), indent=2, ensure_ascii=False) + "\n",
    )

    if any(
        int(row["python_cpp_mismatches"]) != 0
        or int(row["python_sage_mismatches"]) != 0
        for row in rows
    ):
        raise SystemExit("benchmark backend mismatches detected")


if __name__ == "__main__":
    main()
