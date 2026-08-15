from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterator, Mapping


CASE_ORDER = ("A1", "A2", "A3", "A4", "A5", "A6")

_INTEGER_KEYS = (
    "seed",
    "a_min",
    "a_max",
    "b_min",
    "b_max",
    "sample_size",
    "reference_total",
    "calibration_a_min",
    "calibration_a_max",
    "calibration_b_min",
    "calibration_b_max",
)
_BOOLEAN_KEYS = ("include_zero_coefficients",)
_EXPECTED_KEYS = frozenset((*_INTEGER_KEYS, *_BOOLEAN_KEYS))


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    seed: int
    a_min: int
    a_max: int
    b_min: int
    b_max: int
    sample_size: int
    reference_total: int
    include_zero_coefficients: bool
    calibration_a_min: int
    calibration_a_max: int
    calibration_b_min: int
    calibration_b_max: int


def _validate_config(config: ExperimentConfig) -> None:
    ordered_ranges = (
        ("a_min", config.a_min, "a_max", config.a_max),
        ("b_min", config.b_min, "b_max", config.b_max),
        (
            "calibration_a_min",
            config.calibration_a_min,
            "calibration_a_max",
            config.calibration_a_max,
        ),
        (
            "calibration_b_min",
            config.calibration_b_min,
            "calibration_b_max",
            config.calibration_b_max,
        ),
    )
    for minimum_name, minimum, maximum_name, maximum in ordered_ranges:
        if minimum > maximum:
            raise ValueError(
                f"{minimum_name} must be less than or equal to {maximum_name}"
            )
    for name, value in (
        ("sample_size", config.sample_size),
        ("reference_total", config.reference_total),
    ):
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if not config.include_zero_coefficients:
        if config.a_min == config.a_max == 0:
            raise ValueError("a range cannot produce a nonzero coefficient")
        if config.b_min == config.b_max == 0:
            raise ValueError("b range cannot produce a nonzero coefficient")


def _parse_scalar(key: str, value: str) -> int | bool:
    if key in _BOOLEAN_KEYS:
        if value == "true":
            return True
        if value == "false":
            return False
        raise ValueError(f"invalid boolean value for {key}: {value}")
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"invalid integer value for {key}: {value}") from error


def load_config(path: Path) -> ExperimentConfig:
    values: dict[str, int | bool] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"invalid configuration line {line_number}")
        key, raw_value = (part.strip() for part in line.split(":", maxsplit=1))
        if key not in _EXPECTED_KEYS:
            raise ValueError(f"unknown configuration key: {key}")
        if key in values:
            raise ValueError(f"duplicate configuration key: {key}")
        if not raw_value:
            raise ValueError(f"missing scalar value for {key}")
        values[key] = _parse_scalar(key, raw_value)

    for key in (*_INTEGER_KEYS, *_BOOLEAN_KEYS):
        if key not in values:
            raise ValueError(f"missing configuration key: {key}")

    config = ExperimentConfig(**values)  # type: ignore[arg-type]
    _validate_config(config)
    return config


def generate_candidate_pairs(
    config: ExperimentConfig,
    count: int | None = None,
) -> Iterator[tuple[int, int]]:
    requested = config.sample_size if count is None else count
    if requested < 0:
        raise ValueError("candidate count must be nonnegative")
    random_source = random.Random(config.seed)
    generated = 0
    while generated < requested:
        a = random_source.randint(config.a_min, config.a_max)
        b = random_source.randint(config.b_min, config.b_max)
        if not config.include_zero_coefficients and (a == 0 or b == 0):
            continue
        yield a, b
        generated += 1


def allocate_reference_quota(
    available_per_case: Mapping[str, int],
    reference_total: int,
    limit_per_case: int | None = None,
) -> dict[str, int]:
    if reference_total < 0:
        raise ValueError("reference_total must be nonnegative")
    if limit_per_case is not None and limit_per_case < 0:
        raise ValueError("limit_per_case must be nonnegative")
    unknown = set(available_per_case) - set(CASE_ORDER)
    if unknown:
        raise ValueError(f"unknown cases: {', '.join(sorted(unknown))}")

    available = {
        case: min(
            max(0, int(available_per_case.get(case, 0))),
            limit_per_case if limit_per_case is not None else reference_total,
        )
        for case in CASE_ORDER
    }
    selected = {case: 0 for case in CASE_ORDER}

    remaining_total = min(reference_total, sum(available.values()))
    active = [case for case in CASE_ORDER if available[case] > 0]
    while remaining_total > 0 and active:
        share, remainder = divmod(remaining_total, len(active))
        progress = 0
        for index, case in enumerate(active):
            target = share + (1 if index < remainder else 0)
            room = available[case] - selected[case]
            addition = min(target, room)
            selected[case] += addition
            remaining_total -= addition
            progress += addition
        active = [case for case in active if selected[case] < available[case]]
        if progress == 0:
            break
    return selected
