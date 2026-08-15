from __future__ import annotations

from dataclasses import dataclass, field
from math import isqrt
from typing import Any

from .curve import RationalCurve


ALLOWED_TORSION_TYPES_BY_CASE: dict[str, list[str]] = {
    "A1": ["Z/3Z", "Z/9Z"],
    "A2": ["Z/6Z", "Z/12Z"],
    "A3": ["Z/2Z x Z/6Z"],
    "A4": ["{O}", "Z/5Z", "Z/7Z"],
    "A5": ["Z/2Z", "Z/4Z", "Z/8Z", "Z/10Z"],
    "A6": ["Z/2Z x Z/2Z", "Z/2Z x Z/4Z", "Z/2Z x Z/8Z"],
}


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    curve: RationalCurve
    discriminant_nonzero: bool
    c2: int
    has_3_torsion_indicator: bool
    case: str
    allowed_torsion_types: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _positive_divisors(n: int) -> list[int]:
    n = abs(n)
    if n == 0:
        return [0]
    divisors: list[int] = []
    for candidate in range(1, isqrt(n) + 1):
        if n % candidate == 0:
            divisors.append(candidate)
            other = n // candidate
            if other != candidate:
                divisors.append(other)
    return sorted(divisors)


def _integer_root_candidates_for_constant(constant: int) -> list[int]:
    if constant == 0:
        return [0]
    candidates: list[int] = []
    for value in _positive_divisors(constant):
        candidates.extend([-value, value])
    return sorted(set(candidates))


def _exact_integer_cube_root(n: int) -> int | None:
    if n == 0:
        return 0
    sign = -1 if n < 0 else 1
    target = abs(n)
    low = 0
    high = 1
    while high**3 < target:
        high *= 2
    while low <= high:
        mid = (low + high) // 2
        cube = mid**3
        if cube == target:
            return sign * mid
        if cube < target:
            low = mid + 1
        else:
            high = mid - 1
    return None


def rational_2_torsion_roots(curve: RationalCurve) -> list[int]:
    candidates = set(_integer_root_candidates_for_constant(curve.b))
    if curve.b == 0 and -curve.a >= 0:
        root = isqrt(-curve.a)
        if root * root == -curve.a:
            candidates.update({-root, root})
    return sorted(x for x in candidates if curve.cubic(x) == 0)


def _is_square(n: int) -> bool:
    if n < 0:
        return False
    root = isqrt(n)
    return root * root == n


def integer_3_torsion_candidates(curve: RationalCurve) -> list[dict[str, int]]:
    candidates: list[dict[str, int]] = []
    if curve.a != 0:
        x_candidates = _integer_root_candidates_for_constant(-(curve.a**2))
    else:
        x_candidates = [0]
        cubic_root = _exact_integer_cube_root(-4 * curve.b)
        if cubic_root is not None:
            x_candidates.append(cubic_root)
    for x in x_candidates:
        if curve.third_division_polynomial(x) != 0:
            continue
        y_squared = curve.cubic(x)
        if y_squared == 0 or not _is_square(y_squared):
            continue
        candidates.append({"x": x, "y_abs": isqrt(y_squared)})
    return candidates


def classify_a1_a6(curve: RationalCurve) -> ClassificationResult:
    discriminant_nonzero = curve.is_nonsingular()
    if not discriminant_nonzero:
        raise ValueError(f"Cannot classify singular curve {curve.short_name}")

    roots_2 = rational_2_torsion_roots(curve)
    c2 = len(roots_2)
    candidates_3 = integer_3_torsion_candidates(curve)
    has_3 = bool(candidates_3)

    if c2 not in {0, 1, 3}:
        raise ValueError(f"Unexpected 2-torsion root count {c2} for {curve.short_name}")

    if has_3 and c2 == 0:
        case = "A1"
    elif has_3 and c2 == 1:
        case = "A2"
    elif has_3 and c2 == 3:
        case = "A3"
    elif not has_3 and c2 == 0:
        case = "A4"
    elif not has_3 and c2 == 1:
        case = "A5"
    else:
        case = "A6"

    return ClassificationResult(
        curve=curve,
        discriminant_nonzero=discriminant_nonzero,
        c2=c2,
        has_3_torsion_indicator=has_3,
        case=case,
        allowed_torsion_types=ALLOWED_TORSION_TYPES_BY_CASE[case],
        diagnostics={
            "two_torsion_roots": roots_2,
            "three_torsion_candidates": candidates_3,
        },
    )
