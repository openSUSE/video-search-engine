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
