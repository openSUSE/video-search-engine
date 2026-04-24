from pathlib import Path

import yaml


def test_local_env_files_are_ignored_without_hiding_example_file():
    for ignore_file in [".gitignore", ".dockerignore"]:
        patterns = Path(ignore_file).read_text().splitlines()

        assert ".env.*" in patterns
        assert "!.env.example" in patterns


def test_compose_services_have_healthchecks():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())

    for service_name in ["api", "chroma", "rabbitmq", "ui-search", "ui-speaker"]:
        assert "healthcheck" in compose["services"][service_name]


def test_compose_dependencies_wait_for_healthy_services():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())

    assert compose["services"]["api"]["depends_on"]["chroma"]["condition"] == "service_healthy"
    assert compose["services"]["ui-search"]["depends_on"]["api"]["condition"] == "service_healthy"
    assert (
        compose["services"]["ingestion-worker"]["depends_on"]["chroma"]["condition"]
        == "service_healthy"
    )
    assert (
        compose["services"]["ingestion-worker"]["depends_on"]["rabbitmq"]["condition"]
        == "service_healthy"
    )
