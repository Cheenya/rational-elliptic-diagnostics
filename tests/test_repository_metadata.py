from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import subprocess

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TITLE = (
    "Воспроизводимый вычислительный конвейер предварительной диагностики "
    "рациональных эллиптических кривых"
)
METADATA_FILES = (
    "README.md",
    "LICENSE-CODE",
    "LICENSE-DATA",
    "CITATION.cff",
    "environment.yml",
    ".gitignore",
)


def _load_yaml(name: str) -> object:
    completed = subprocess.run(
        [
            "sage",
            "-python",
            "-c",
            (
                "import json, pathlib, sys, yaml; "
                "data = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8')); "
                "print(json.dumps(data, ensure_ascii=False))"
            ),
            str(REPOSITORY_ROOT / name),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _author_names(authors: object) -> list[tuple[str, str]]:
    assert isinstance(authors, list)
    return [
        (author["family-names"], author["given-names"])
        for author in authors
    ]


def _manifest_entries() -> list[tuple[str, str]]:
    manifest = (REPOSITORY_ROOT / "MANIFEST.sha256").read_text(encoding="utf-8")
    entries: list[tuple[str, str]] = []
    for line in manifest.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  \./(.+)", line)
        assert match is not None, f"invalid manifest entry: {line!r}"
        entries.append((match.group(1), match.group(2)))
    return entries


def test_required_repository_metadata_files_exist() -> None:
    missing = [name for name in METADATA_FILES if not (REPOSITORY_ROOT / name).is_file()]
    assert missing == []


def test_citation_metadata_has_exact_identity_and_factual_references() -> None:
    citation = _load_yaml("CITATION.cff")
    assert isinstance(citation, dict)
    assert citation["cff-version"] == "1.2.0"
    assert citation["type"] == "software"
    assert citation["title"] == TITLE
    assert citation["version"] == "0.1.0"
    assert citation["license"] == "MIT"
    assert str(citation["date-released"]).startswith("2026-")
    expected_authors = [
        ("Чистяков", "Никита Андреевич"),
        ("Адамова", "Раиса Сергеевна"),
    ]
    assert _author_names(citation["authors"]) == expected_authors

    preferred = citation["preferred-citation"]
    assert preferred["title"] == TITLE
    assert preferred["year"] == 2026
    assert _author_names(preferred["authors"]) == expected_authors
    assert "doi" not in preferred
    assert "url" not in preferred

    references = citation["references"]
    serialized = json.dumps(references, ensure_ascii=False, sort_keys=True)
    for identifier in (
        "978-5-6054088-7-1",
        "10.1515/crll.1937.177.238",
        "10.1007/BF02684339",
        "10.1007/978-0-387-09494-6",
        "10.5281/zenodo.8042260",
        "https://www.sagemath.org",
    ):
        assert identifier in serialized
    assert "Вычисление группы точек конечного порядка рациональной эллиптической кривой" in serialized
    assert "10.8" in serialized


def test_citation_metadata_has_joint_article_as_separate_reference() -> None:
    citation = _load_yaml("CITATION.cff")
    assert isinstance(citation, dict)
    title = (
        "Вычислительная методика быстрой проверки наличия точек 3-го порядка "
        "на рациональных эллиптических кривых"
    )
    matches = [
        reference
        for reference in citation["references"]
        if reference.get("title") == title
    ]
    assert matches == [
        {
            "type": "article",
            "title": title,
            "authors": [
                {
                    "family-names": "Чистяков",
                    "given-names": "Никита Андреевич",
                },
                {
                    "family-names": "Адамова",
                    "given-names": "Раиса Сергеевна",
                },
            ],
            "journal": (
                "Вестник Воронежского государственного университета. "
                "Серия: Физика. Математика"
            ),
            "year": 2026,
            "issue": 1,
            "start": 59,
            "end": 67,
        }
    ]


def test_environment_pins_reproduction_toolchain() -> None:
    environment = _load_yaml("environment.yml")
    assert isinstance(environment, dict)
    assert environment["name"] == "rational-elliptic-diagnostics"
    assert environment["channels"] == ["conda-forge"]
    assert environment["dependencies"] == [
        "python=3.13",
        "sage=10.8",
        "cmake=4.2",
        "pytest=9",
        "cxx-compiler=1.11",
        "pip",
    ]


def test_readme_states_scope_commands_results_and_measurement_contract() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.startswith(f"# {TITLE}\n")
    for required in (
        "y^2 = x^3 + ax + b",
        "Stage A",
        "A1–A6",
        "Stage B",
        "torsion_subgroup()",
        "SageMath 10.8",
        "Sage Python 3.13.7",
        "CMake 4.2.3",
        "C++17",
        "Apple clang 21.0.0",
        "__int128",
        "Clang или GCC",
        "conda env create -f environment.yml",
        "python -m pytest -q",
        "python scripts/run_stage_a.py --config configs/conference.yml --output-dir output",
        "sage -python scripts/run_sage_reference.py --config configs/conference.yml --input output/stage_a_rows.csv --output-dir output",
        "python scripts/verify_results.py --config configs/conference.yml --input-dir output --results-dir results",
        "sage -python scripts/run_benchmark.py --config configs/conference.yml --sizes 10000 30000 100000 300000 --sample-per-size 1000 --repeats 3 --output-dir results",
        "sage -python scripts/run_workflow_benchmark.py --config configs/conference.yml --n 200000 --k 2000 --repeats 5 --output-dir results",
        "results/stage_a_summary.csv",
        "results/stage_b_reference.csv",
        "results/calibration.csv",
        "results/benchmark_scaling.csv",
        "results/workflow_benchmark_runs.csv",
        "results/workflow_benchmark_summary.csv",
        "results/environment.json",
        "2 000",
        "200 000",
        "300+300",
        "точное покрытие",
        "100%",
        "1%",
        "20,4",
        "зависят от машины и методики",
        "MIT",
        "CC BY 4.0",
    ):
        assert required in readme
    assert "одинаковый выход двух гибридных сценариев" in readme


def test_readme_links_article_and_poster_with_exact_attribution() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    heading = "## Связь со статьёй и плакатом"
    assert readme.count(heading) == 1
    section = readme.split(heading, maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    for required in (
        "статья является методической основой работы",
        "Вестник Воронежского государственного университета. Серия: Физика. Математика",
        "текущему снимку репозитория",
        "Докладчик по плакату — только Н. А. Чистяков",
        "Р. С. Адамова указана как соавтор публикации и научной основы",
    ):
        assert required in section


def test_manifest_entries_are_sorted_exist_and_match_sha256() -> None:
    entries = _manifest_entries()
    relative_paths = [relative_path for _, relative_path in entries]
    assert relative_paths == sorted(relative_paths)
    assert "MANIFEST.sha256" not in relative_paths

    for expected_digest, relative_path in entries:
        path = REPOSITORY_ROOT / relative_path
        assert path.is_file(), f"missing manifest path: {relative_path}"
        actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_digest == expected_digest, f"stale manifest digest: {relative_path}"


def test_manifest_entries_match_git_tracked_files_when_checkout_available() -> None:
    try:
        checkout = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        pytest.skip("git is unavailable; archive hashes remain covered")

    if checkout.returncode != 0:
        pytest.skip("source archive without Git metadata")
    if Path(checkout.stdout.strip()).resolve() != REPOSITORY_ROOT.resolve():
        pytest.skip("source archive is nested in a different Git checkout")

    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8").split("\0")
    expected_paths = {path for path in tracked if path and path != "MANIFEST.sha256"}
    manifest_paths = {relative_path for _, relative_path in _manifest_entries()}
    assert manifest_paths == expected_paths


def test_license_split_uses_exact_identifiers_and_canonical_terms() -> None:
    code_license = (REPOSITORY_ROOT / "LICENSE-CODE").read_text(encoding="utf-8")
    assert "SPDX-License-Identifier: MIT" in code_license
    assert "Copyright (c) 2026 Nikita Chistyakov" in code_license
    assert "Permission is hereby granted, free of charge" in code_license
    assert "THE SOFTWARE IS PROVIDED \"AS IS\"" in code_license

    data_license = (REPOSITORY_ROOT / "LICENSE-DATA").read_text(encoding="utf-8")
    assert "SPDX-License-Identifier: CC-BY-4.0" in data_license
    assert "Nikita Chistyakov and Raisa Adamova" in data_license
    assert "https://creativecommons.org/licenses/by/4.0/legalcode" in data_license


@pytest.mark.parametrize(
    "path",
    (
        ".pytest_cache/state",
        "src/elliptic_diagnostics/__pycache__/classifier.cpython-313.pyc",
        ".venv/bin/python",
        "build/CMakeCache.txt",
        "output/stage_a_rows.csv",
        ".DS_Store",
        "src/rational_elliptic_diagnostics.egg-info/PKG-INFO",
    ),
)
def test_gitignore_ignores_only_generated_artifact_classes(path: str) -> None:
    completed = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", path],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    assert completed.returncode == 0


@pytest.mark.parametrize(
    "path",
    (
        "results/stage_a_summary.csv",
        "src/elliptic_diagnostics/classifier.py",
        "tests/test_classifier.py",
        "configs/conference.yml",
        "data/fixtures/torsion_examples.csv",
        "MANIFEST.sha256",
    ),
)
def test_gitignore_keeps_research_sources_and_results(path: str) -> None:
    completed = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", path],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    assert completed.returncode == 1


def test_public_metadata_has_no_placeholders_contacts_or_internal_markers() -> None:
    citation = _load_yaml("CITATION.cff")
    assert isinstance(citation, dict)
    assert not {
        "email",
        "orcid",
        "repository",
        "repository-artifact",
        "repository-code",
    }.intersection(citation)
    assert not {
        "email",
        "orcid",
        "repository",
        "repository-artifact",
        "repository-code",
        "url",
    }.intersection(citation["preferred-citation"])

    combined = "\n".join(
        (REPOSITORY_ROOT / name).read_text(encoding="utf-8")
        for name in METADATA_FILES
    )
    lowered = combined.lower()
    for forbidden in (
        "placeholder",
        "todo",
        "tbd",
        "msvc",
    ):
        assert forbidden not in lowered
    assert re.search(r"/(?:Users|home)/[^/\s]+", combined, flags=re.IGNORECASE) is None
    assert re.search(r"[A-Z]:\\", combined) is None
    assert re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", combined) is None
    assert set(re.findall(r"https://[^\s\"']+", combined)) == {
        "https://creativecommons.org/licenses/by/4.0/legalcode",
        "https://www.sagemath.org",
    }
