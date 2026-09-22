from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]


def test_uv_is_the_only_repository_environment_manager():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")

    assert "[dependency-groups]" in pyproject
    assert "environments = [\"sys_platform == 'linux'\"]" in pyproject
    assert 'name = "vision-lens"' in lock
    assert not (REPO_ROOT / "environment.yml").exists()
