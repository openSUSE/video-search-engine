import ast
import re
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


def test_kubernetes_configmap_exposes_runtime_collection_name():
    configmap = yaml.safe_load(Path("k8s/configmap.yaml").read_text())

    assert configmap["data"]["CHROMA_COLLECTION"] == "video_search_engine"


def test_service_dockerfiles_use_pinned_python_base_image():
    for dockerfile in Path("docker").glob("*.Dockerfile"):
        first_line = dockerfile.read_text().splitlines()[0]

        assert first_line == "FROM python:3.12.13-slim"


def test_api_main_defers_heavy_search_dependency_imports():
    tree = ast.parse(Path("api/main.py").read_text())
    imported_modules = set()

    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split(".")[0])

    assert "chromadb" not in imported_modules
    assert "sentence_transformers" not in imported_modules


def test_chroma_runtime_images_are_pinned():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    k8s_chroma = list(yaml.safe_load_all(Path("k8s/chroma.yaml").read_text()))[0]

    compose_image = compose["services"]["chroma"]["image"]
    k8s_image = k8s_chroma["spec"]["template"]["spec"]["containers"][0]["image"]

    assert compose_image.startswith("chromadb/chroma:${CHROMA_IMAGE_TAG:-")
    assert k8s_image.startswith("chromadb/chroma:")
    assert ":latest" not in compose_image
    assert ":latest" not in k8s_image


def test_rabbitmq_runtime_images_are_pinned():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    k8s_rabbitmq = list(yaml.safe_load_all(Path("k8s/rabbitmq.yaml").read_text()))[0]

    compose_image = compose["services"]["rabbitmq"]["image"]
    k8s_image = k8s_rabbitmq["spec"]["template"]["spec"]["containers"][0]["image"]

    compose_tag_match = re.search(r"\$\{RABBITMQ_IMAGE_TAG:-(.+)\}", compose_image)
    assert compose_tag_match is not None

    compose_default_tag = compose_tag_match.group(1)
    k8s_tag = k8s_image.removeprefix("rabbitmq:")

    assert compose_image.startswith("rabbitmq:${RABBITMQ_IMAGE_TAG:-")
    assert re.match(r"^\d+\.\d+\.\d+-management$", compose_default_tag)
    assert re.match(r"^\d+\.\d+\.\d+-management$", k8s_tag)
    assert compose_default_tag == k8s_tag
    assert ":latest" not in compose_image
    assert ":latest" not in k8s_image
