from __future__ import annotations

import argparse
import csv
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from elliptic_diagnostics.curve import RationalCurve
from elliptic_diagnostics.experiment import CASE_ORDER, generate_candidate_pairs, load_config


CSV_COLUMNS = (
    "sample_id",
    "a",
    "b",
    "discriminant",
    "c2",
    "has3",
    "case",
    "three_torsion_x",
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


def _build_classifier(output_dir: Path) -> Path:
    build_dir = output_dir / "cpp-build"
    subprocess.run(
        ["cmake", "-S", str(REPOSITORY_ROOT / "cpp"), "-B", str(build_dir)],
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


def _classify(executable: Path, candidates: list[tuple[str, int, int]]) -> list[list[str]]:
    classifier_input = "".join(
        f"{sample_id}\t{a}\t{b}\n"
        for sample_id, a, b in candidates
    )
    completed = subprocess.run(
        [str(executable)],
        input=classifier_input,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.split("\t") for line in completed.stdout.splitlines()]
    if len(rows) != len(candidates) or any(len(row) != 5 for row in rows):
        raise RuntimeError("C++ classifier returned an invalid row count or schema")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", default="output")
    arguments = parser.parse_args()

    config = load_config(_root_relative(arguments.config))
    requested = config.sample_size if arguments.limit is None else arguments.limit
    if requested < 0:
        parser.error("--limit must be nonnegative")
    output_dir = _root_relative(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        (f"sample_{index:09d}", a, b)
        for index, (a, b) in enumerate(
            generate_candidate_pairs(config, requested),
            start=1,
        )
    ]
    classified = _classify(_build_classifier(output_dir), candidates)

    case_counts = {case: 0 for case in CASE_ORDER}
    singular_excluded = 0
    csv_buffer = StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for (sample_id, a, b), result in zip(candidates, classified, strict=True):
        returned_id, c2, has3, case, three_torsion_x = result
        if returned_id != sample_id:
            raise RuntimeError("C++ classifier changed sample row order")
        curve = RationalCurve(a=a, b=b)
        if not curve.is_nonsingular():
            singular_excluded += 1
            continue
        if case not in case_counts:
            raise RuntimeError(f"C++ classifier returned invalid case: {case}")
        case_counts[case] += 1
        writer.writerow(
            {
                "sample_id": sample_id,
                "a": a,
                "b": b,
                "discriminant": curve.discriminant,
                "c2": c2,
                "has3": has3,
                "case": case,
                "three_torsion_x": three_torsion_x,
            }
        )

    nonsingular_rows = requested - singular_excluded
    metadata = {
        "backend": "cpp17-cmake",
        "case_counts": case_counts,
        "include_zero_coefficients": config.include_zero_coefficients,
        "input_pairs": requested,
        "nonsingular_rows": nonsingular_rows,
        "seed": config.seed,
        "singular_excluded": singular_excluded,
    }
    _write_text_atomic(output_dir / "stage_a_rows.csv", csv_buffer.getvalue())
    _write_text_atomic(
        output_dir / "stage_a_metadata.json",
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    )


if __name__ == "__main__":
    main()
