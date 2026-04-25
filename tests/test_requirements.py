from pathlib import Path

import pytest
from pip._vendor.packaging.requirements import InvalidRequirement, Requirement


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

    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r ", "--", "git+")):
            continue

        try:
            Requirement(line)
        except InvalidRequirement as exc:
            raise AssertionError(f"{path}:{line_number} has invalid requirement syntax: {line}") from exc


def test_runtime_requirements_do_not_include_unused_pandas_dependency():
    for requirements_file in ["requirements-ui.txt", "requirements-ingestion.txt"]:
        direct_requirements = {
            line.strip().split(";", maxsplit=1)[0].strip()
            for line in Path(requirements_file).read_text().splitlines()
            if line.strip() and not line.strip().startswith(("#", "-"))
        }

        assert "pandas" not in direct_requirements
