import pytest

from elliptic_diagnostics import (
    RationalCurve,
    classify_a1_a6,
    integer_3_torsion_candidates,
    rational_2_torsion_roots,
)


@pytest.mark.parametrize(
    ("a", "b", "expected_case", "expected_c2", "expected_three_indicator"),
    [
        (-21, 37, "A1", 0, True),
        (-15, 22, "A2", 1, True),
        (-24003, 1296702, "A3", 3, True),
        (-30, -80, "A4", 0, False),
        (-30, -63, "A5", 1, False),
        (-28, -48, "A6", 3, False),
    ],
)
def test_classifier_assigns_each_a1_a6_case(
    a: int,
    b: int,
    expected_case: str,
    expected_c2: int,
    expected_three_indicator: bool,
) -> None:
    result = classify_a1_a6(RationalCurve(a=a, b=b))

    assert result.case == expected_case
    assert result.c2 == expected_c2
    assert result.has_3_torsion_indicator is expected_three_indicator


def test_rational_two_torsion_roots_returns_all_integer_roots() -> None:
    curve = RationalCurve(a=-24003, b=1296702)

    assert rational_2_torsion_roots(curve) == [-177, 66, 111]


def test_integer_three_torsion_candidates_returns_integral_point_data() -> None:
    curve = RationalCurve(a=-21, b=37)

    assert integer_3_torsion_candidates(curve) == [{"x": 3, "y_abs": 1}]
