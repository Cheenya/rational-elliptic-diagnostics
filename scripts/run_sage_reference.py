from __future__ import annotations

import argparse
import csv
from io import StringIO
import json
from pathlib import Path
import random
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from elliptic_diagnostics.experiment import CASE_ORDER, allocate_reference_quota, load_config
from sage.all import EllipticCurve, QQ


CSV_COLUMNS = (
    "sample_id",
    "a",
    "b",
    "case",
    "torsion_order",
    "torsion_invariants",
    "generators",
)


def _root_relative(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


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


def _load_baskets(input_path: Path) -> dict[str, list[dict[str, str]]]:
    baskets: dict[str, list[dict[str, str]]] = {case: [] for case in CASE_ORDER}
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "a", "b", "case"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Stage A input is missing required columns")
        for row in reader:
            case = row["case"]
            if case not in baskets:
                raise ValueError(f"invalid Stage A case: {case}")
            baskets[case].append(row)
    return baskets


def _serialize_invariants(torsion) -> str:
    return json.dumps([int(value) for value in torsion.invariants()], separators=(",", ":"))


def _serialize_generators(torsion) -> str:
    coordinates = [
        [str(coordinate) for coordinate in generator.element()]
        for generator in torsion.gens()
    ]
    return json.dumps(coordinates, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit-per-case", type=int)
    parser.add_argument("--input")
    parser.add_argument("--output-dir", default="output")
    arguments = parser.parse_args()

    if arguments.limit_per_case is not None and arguments.limit_per_case < 0:
        parser.error("--limit-per-case must be nonnegative")
    config = load_config(_root_relative(arguments.config))
    output_dir = _root_relative(arguments.output_dir)
    input_path = (
        _root_relative(arguments.input)
        if arguments.input is not None
        else output_dir / "stage_a_rows.csv"
    )
    baskets = _load_baskets(input_path)
    available = {case: len(baskets[case]) for case in CASE_ORDER}
    selected_per_case = allocate_reference_quota(
        available,
        config.reference_total,
        arguments.limit_per_case,
    )

    random_source = random.Random(config.seed)
    selected: list[dict[str, str]] = []
    for case in CASE_ORDER:
        candidates = list(baskets[case])
        random_source.shuffle(candidates)
        selected.extend(candidates[: selected_per_case[case]])
    selected.sort(key=lambda row: (CASE_ORDER.index(row["case"]), row["sample_id"]))

    csv_buffer = StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in selected:
        a = int(row["a"])
        b = int(row["b"])
        curve = EllipticCurve(QQ, [0, 0, 0, a, b])
        torsion = curve.torsion_subgroup()
        writer.writerow(
            {
                "sample_id": row["sample_id"],
                "a": a,
                "b": b,
                "case": row["case"],
                "torsion_order": int(torsion.order()),
                "torsion_invariants": _serialize_invariants(torsion),
                "generators": _serialize_generators(torsion),
            }
        )

    selection = {
        "available_per_case": available,
        "limit_per_case": arguments.limit_per_case,
        "reference_total": config.reference_total,
        "seed": config.seed,
        "selected_per_case": selected_per_case,
        "selected_total": len(selected),
    }
    _write_text_atomic(output_dir / "stage_b_reference_rows.csv", csv_buffer.getvalue())
    _write_text_atomic(
        output_dir / "stage_b_selection.json",
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
    )


if __name__ == "__main__":
    main()
