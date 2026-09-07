import importlib.util
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]

EXAMPLE_SCRIPTS = {
    "run_dino_vits8_attention.py": "vit_attention.example.yaml",
    "run_dinov2_reg4_attention.py": "vit_attention.dinov2_reg4.example.yaml",
    "run_dinov2_reg4_rollout.py": "vit_rollout.dinov2_reg4.example.yaml",
    "run_resnet50_gradcam.py": "gradcam.example.yaml",
    "run_dinov2_pca.py": "patch_pca.dinov2.example.yaml",
}


def test_each_example_config_has_a_matching_script():
    scripts_dir = REPO_ROOT / "scripts"

    for script_name, config_name in EXAMPLE_SCRIPTS.items():
        script = scripts_dir / script_name
        assert script.is_file()
        assert config_name in script.read_text(encoding="utf-8")


def test_example_scripts_forward_cli_overrides_without_running_models():
    for script_name in EXAMPLE_SCRIPTS:
        script_path = REPO_ROOT / "scripts" / script_name
        spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        received = []
        module.cli_main = partial(_record_args, received)

        assert module.main(["--set", "runtime.device=cpu"]) == 0
        assert received == [
            "--config",
            str(module.CONFIG),
            "--set",
            "runtime.device=cpu",
        ]


def _record_args(received, argv):
    received.extend(argv)
    return 0
