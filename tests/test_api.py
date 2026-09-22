from pathlib import Path
from types import SimpleNamespace

import pytest

import vision_lens
from vision_lens.errors import ConfigurationError, PipelineError


def test_public_api_uses_specific_configuration_errors():
    with pytest.raises(ConfigurationError, match="raw_config must be a mapping"):
        vision_lens.parse_config([])  # type: ignore[arg-type]


def test_public_api_wraps_pipeline_failures(monkeypatch, tmp_path):
    from vision_lens import pipeline

    source = tmp_path / "input.jpg"
    source.touch()
    config = vision_lens.parse_config(
        {
            "input": {"files": [str(source)]},
            "model": {
                "architecture": "vit",
                "backend": "timm",
                "name": "mock_vit",
            },
            "analysis": {"method": "attention", "layers": [0]},
            "output": {"directory": str(tmp_path / "results")},
        }
    )

    def fail(_config):
        raise ValueError("model rejected the request")

    monkeypatch.setattr(pipeline, "_dispatch_pipeline", fail)

    with pytest.raises(PipelineError, match="model rejected") as error:
        vision_lens.run_pipeline_from_config(config, show_progress=False)

    assert isinstance(error.value.__cause__, ValueError)


def test_run_result_describes_the_common_result_surface():
    result = SimpleNamespace(config=object(), output_paths=(Path("output.png"),))

    assert isinstance(result, vision_lens.RunResult)


def test_pipeline_module_exports_only_generic_runners():
    from vision_lens import pipeline

    assert pipeline.__all__ == ["run_pipeline", "run_pipeline_from_config"]
