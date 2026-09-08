from pathlib import Path

import yaml

from vision_lens.config import load_config

REPO_ROOT = Path(__file__).parents[1]


def test_documented_example_configs_validate_without_loading_models():
    for path in sorted((REPO_ROOT / "configs").glob("*.example.yaml")):
        config = load_config(path)
        assert config.input.paths


def test_readme_has_result_placeholders_without_committed_gallery_media():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "Results gallery placeholder" in readme
    assert "PCA result placeholder" in readme
    assert "Video demo placeholder" in readme
    assert not (REPO_ROOT / "docs/assets/gallery").exists()


def test_conda_environment_installs_project_extras_from_pyproject():
    environment = yaml.safe_load(
        (REPO_ROOT / "environment.yml").read_text(encoding="utf-8")
    )

    assert environment["name"] == "vision-lens"
    assert "conda-forge" in environment["channels"]
    assert "-e .[dev,video]" in environment["dependencies"][-1]["pip"]
