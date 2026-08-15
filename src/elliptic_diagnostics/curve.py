from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RationalCurve:
    a: int
    b: int

    def __post_init__(self) -> None:
        if type(self.a) is not int or type(self.b) is not int:
            raise TypeError("RationalCurve requires integer coefficients a and b")

    @property
    def short_name(self) -> str:
        return f"E({self.a},{self.b})"

    @property
    def discriminant_test_value(self) -> int:
        return 4 * self.a**3 + 27 * self.b**2

    @property
    def discriminant(self) -> int:
        return -16 * self.discriminant_test_value

    def is_nonsingular(self) -> bool:
        return self.discriminant_test_value != 0

    def cubic(self, x: int) -> int:
        return x**3 + self.a * x + self.b

    def third_division_polynomial(self, x: int) -> int:
        return 3 * x**4 + 6 * self.a * x**2 + 12 * self.b * x - self.a**2
