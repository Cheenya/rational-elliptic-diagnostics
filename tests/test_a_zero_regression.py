from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

import pytest

from elliptic_diagnostics import RationalCurve, classify_a1_a6


@dataclass(frozen=True, slots=True)
class CppClassificationResult:
    case: str
    c2: int
    has3: bool
    three_torsion_x: list[int]


@pytest.fixture(scope="module")
def cpp_classifier(tmp_path_factory: pytest.TempPathFactory):
    repository_root = Path(__file__).resolve().parents[1]
    build_dir = tmp_path_factory.mktemp("phase-a-build")

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

    executable_candidates = [
        build_dir / "phase_a",
        build_dir / "Release" / "phase_a",
    ]
    executable = next(path for path in executable_candidates if path.is_file())

    def classify(curve: RationalCurve) -> CppClassificationResult:
        completed = subprocess.run(
            [str(executable)],
            input=f"regression\t{curve.a}\t{curve.b}\n",
            check=True,
            capture_output=True,
            text=True,
        )
        sample_id, c2, has3, case, roots = completed.stdout.rstrip("\n").split("\t")
        assert sample_id == "regression"
        return CppClassificationResult(
            case=case,
            c2=int(c2),
            has3=has3 == "1",
            three_torsion_x=[] if roots == "-" else [int(x) for x in roots.split(",")],
        )

    return classify


def test_a_zero_cube_root_branch_matches_python(cpp_classifier) -> None:
    curve = RationalCurve(a=0, b=-27648)

    python_result = classify_a1_a6(curve)
    cpp_result = cpp_classifier(curve)

    assert python_result.case == "A1"
    assert cpp_result.case == "A1"
    assert cpp_result.c2 == 0
    assert cpp_result.has3 is True
    assert cpp_result.three_torsion_x == [48]
