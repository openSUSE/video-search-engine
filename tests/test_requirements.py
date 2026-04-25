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
