from pathlib import Path

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


def test_uv_is_the_only_repository_environment_manager():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")

    assert "[dependency-groups]" in pyproject
    assert "environments = [\"sys_platform == 'linux'\"]" in pyproject
    assert 'name = "vision-lens"' in lock
    assert not (REPO_ROOT / "environment.yml").exists()
