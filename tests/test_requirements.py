from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement


def _iter_parseable_requirement_lines(path: Path):
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r ", "--", "git+")):
            continue
        yield line_number, line


@pytest.mark.parametrize(
    "requirements_file",
    [
        "requirements-api.txt",
        "requirements-dev.txt",
        "requirements-ingestion.txt",
        "requirements-ui.txt",
        "requirements.txt",
    ],
)
def test_requirement_files_are_parseable(requirements_file):
    path = Path(requirements_file)

    for line_number, line in _iter_parseable_requirement_lines(path):
        try:
            Requirement(line)
        except InvalidRequirement as exc:
            raise AssertionError(f"{path}:{line_number} has invalid requirement syntax: {line}") from exc


def test_runtime_requirements_do_not_include_unused_pandas_dependency():
    for requirements_file in ["requirements-ui.txt", "requirements-ingestion.txt"]:
        direct_requirements = {
            Requirement(line).name.lower()
            for _line_number, line in _iter_parseable_requirement_lines(
                Path(requirements_file)
            )
        }

        assert "pandas" not in direct_requirements


def test_pydantic_requirement_matches_schema_validator_floor():
    """Pydantic must be pinned to a major-2 line that chromadb supports.

    chromadb 1.5.x imports pydantic ≥ 2.10; the previous <2.10 cap shipped
    a fresh build whose pydantic and chromadb pins disagreed. We pin to a
    range that satisfies both: 2.10 ≤ pydantic < 3.
    """
    for requirements_file in ["requirements-api.txt", "requirements-dev.txt"]:
        direct_requirements = {
            Requirement(line).name.lower(): Requirement(line)
            for _line_number, line in _iter_parseable_requirement_lines(
                Path(requirements_file)
            )
        }

        pydantic_requirement = direct_requirements["pydantic"]
        specifier = pydantic_requirement.specifier
        assert pydantic_requirement.specifier.contains("2.10"), (
            f"{requirements_file} pydantic spec {specifier} must allow >=2.10 (chromadb 1.5.x requires it)"
        )
        assert not pydantic_requirement.specifier.contains("3.0.0"), (
            f"{requirements_file} pydantic spec {specifier} must cap below 3.0"
        )


def test_logging_dependency_is_available_to_setup_logging_entrypoints():
    for requirements_file in [
        "requirements-api.txt",
        "requirements-dev.txt",
        "requirements-ingestion.txt",
        "requirements-ui.txt",
    ]:
        direct_requirement_names = {
            Requirement(line).name.lower()
            for _line_number, line in _iter_parseable_requirement_lines(
                Path(requirements_file)
            )
        }

        assert "colorlog" in direct_requirement_names, (
            f"{requirements_file} must install colorlog because shipped entrypoints "
            "import core.logger.setup_logging"
        )
