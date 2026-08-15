from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import subprocess

import pytest

from elliptic_diagnostics import RationalCurve, classify_a1_a6


@dataclass(frozen=True, slots=True)
class CppClassificationResult:
    c2: int
    has3: bool
    case: str


def load_fixtures() -> list[tuple[int, int, str]]:
    fixture_path = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "torsion_examples.csv"
    with fixture_path.open(newline="") as fixture_file:
        reader = csv.DictReader(fixture_file)
        return [
            (int(row["a"]), int(row["b"]), row["expected_case"])
            for row in reader
        ]


@pytest.fixture(scope="module")
def cpp_classifier(tmp_path_factory: pytest.TempPathFactory):
    repository_root = Path(__file__).resolve().parents[1]
    build_dir = tmp_path_factory.mktemp("phase-a-parity-build")

    subprocess.run(
        ["cmake", "-S", str(repository_root / "cpp"), "-B", str(build_dir)],
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

    executable = next(
        path
        for path in (build_dir / "phase_a", build_dir / "Release" / "phase_a")
        if path.is_file()
    )

    def classify(curve: RationalCurve) -> CppClassificationResult:
        completed = subprocess.run(
            [str(executable)],
            input=f"fixture\t{curve.a}\t{curve.b}\n",
            check=True,
            capture_output=True,
            text=True,
        )
        sample_id, c2, has3, case, _ = completed.stdout.rstrip("\n").split("\t")
        assert sample_id == "fixture"
        return CppClassificationResult(c2=int(c2), has3=has3 == "1", case=case)

    return classify


@pytest.mark.parametrize(("a", "b", "expected_case"), load_fixtures())
def test_python_cpp_match_all_torsion_fixtures(
    a: int,
    b: int,
    expected_case: str,
    cpp_classifier,
) -> None:
    curve = RationalCurve(a=a, b=b)
    python_result = classify_a1_a6(curve)
    cpp_result = cpp_classifier(curve)

    assert python_result.case == expected_case
    assert cpp_result.case == expected_case
    assert cpp_result.case == python_result.case
    assert cpp_result.c2 == python_result.c2
    assert cpp_result.has3 == python_result.has_3_torsion_indicator
