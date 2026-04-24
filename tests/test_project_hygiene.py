import ast
import re
from pathlib import Path

import yaml


def _top_level_imports(path: str) -> set[str]:
    tree = ast.parse(Path(path).read_text())
    imported_modules = set()

    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split(".")[0])

    return imported_modules


def _top_level_import_modules(path: str) -> set[str]:
    tree = ast.parse(Path(path).read_text())
    imported_modules = set()

    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    return imported_modules


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


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


def test_search_ui_timeout_is_configured_for_deployments():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    configmap = yaml.safe_load(Path("k8s/configmap.yaml").read_text())
    env_example = Path(".env.example").read_text().splitlines()

    assert "SEARCH_API_TIMEOUT_SECONDS=10" in env_example
    assert (
        compose["services"]["ui-search"]["environment"]["SEARCH_API_TIMEOUT_SECONDS"]
        == "${SEARCH_API_TIMEOUT_SECONDS:-10}"
    )
    assert configmap["data"]["SEARCH_API_TIMEOUT_SECONDS"] == "10"


def test_search_app_handles_request_failures_through_client_boundary():
    tree = ast.parse(Path("app/ui/search_app.py").read_text())
    imported_client_names = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "app.ui.search_client"
        for alias in node.names
    }
    handled_exceptions = {
        _qualified_name(node.type)
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler) and node.type is not None
    }

    assert "RequestException" in imported_client_names
    assert "RequestException" in handled_exceptions
    assert "requests.exceptions.RequestException" not in handled_exceptions


def test_search_streamlit_ui_defers_core_config_import():
    imported_modules = _top_level_import_modules("app/ui/search_app.py")

    assert "core.config" not in imported_modules


def test_speaker_streamlit_ui_uses_support_path_boundary():
    tree = ast.parse(Path("app/ui/speaker_id_tool.py").read_text())
    imported_modules = _top_level_import_modules("app/ui/speaker_id_tool.py")
    imported_support_names = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "app.ui.speaker_support"
        for alias in node.names
    }
    literal_strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "app.ui.speaker_support" in imported_modules
    assert "normalize_speaker_map" in imported_support_names
    assert "transcript_generic.json" not in literal_strings
    assert "speaker_map.json" not in literal_strings


def test_speaker_flask_ui_defers_config_import():
    imported_modules = _top_level_import_modules("app/main.py")

    assert "core.config" not in imported_modules


def test_kubernetes_configmap_exposes_runtime_collection_name():
    configmap = yaml.safe_load(Path("k8s/configmap.yaml").read_text())

    assert configmap["data"]["CHROMA_COLLECTION"] == "video_search_engine"


def test_service_dockerfiles_use_pinned_python_base_image():
    for dockerfile in Path("docker").glob("*.Dockerfile"):
        first_line = dockerfile.read_text().splitlines()[0]

        assert first_line == "FROM python:3.12.13-slim"


def test_validate_compiles_source_trees_instead_of_fixed_file_list():
    makefile = Path("Makefile").read_text()

    assert "-m compileall -q api app core ingestion_pipeline inspect_db.py" in makefile


def test_api_main_defers_heavy_search_dependency_imports():
    imported_modules = _top_level_import_modules("api/main.py")

    assert "chromadb" not in imported_modules
    assert "core.config" not in imported_modules
    assert "sentence_transformers" not in imported_modules


def test_indexing_step_defers_heavy_dependency_imports():
    imported_modules = _top_level_imports("ingestion_pipeline/steps/step_04_indexing.py")

    assert "chromadb" not in imported_modules
    assert "sentence_transformers" not in imported_modules


def test_segmentation_step_defers_heavy_dependency_imports():
    imported_modules = _top_level_import_modules("ingestion_pipeline/steps/step_02_segmentation.py")

    assert "core.config" not in imported_modules
    assert "cv2" not in imported_modules
    assert "numpy" not in imported_modules
    assert "sentence_transformers" not in imported_modules
    assert "sklearn.metrics.pairwise" not in imported_modules


def test_enrichment_step_defers_provider_dependency_imports():
    imported_modules = _top_level_import_modules("ingestion_pipeline/steps/step_03_enrichment.py")

    assert "google.generativeai" not in imported_modules
    assert "google.api_core" not in imported_modules
    assert "requests" not in imported_modules


def test_extraction_step_defers_heavy_dependency_imports():
    imported_modules = _top_level_import_modules("ingestion_pipeline/steps/step_01_extraction.py")

    assert "PIL" not in imported_modules
    assert "core.config" not in imported_modules
    assert "cv2" not in imported_modules
    assert "ingestion_pipeline.utils.metadata_fetcher" not in imported_modules
    assert "numpy" not in imported_modules
    assert "torch" not in imported_modules
    assert "whisperx" not in imported_modules


def test_metadata_fetcher_defers_tmdb_sdk_import():
    imported_modules = _top_level_import_modules("ingestion_pipeline/utils/metadata_fetcher.py")

    assert "tmdbv3api" not in imported_modules


def test_run_pipeline_defers_runtime_dependency_imports():
    imported_modules = _top_level_import_modules("ingestion_pipeline/run_pipeline.py")

    assert "core.config" not in imported_modules
    assert "requests" not in imported_modules
    assert not any(
        module.startswith("ingestion_pipeline.steps.") for module in imported_modules
    )


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
