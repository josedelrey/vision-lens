from __future__ import annotations

import random
from typing import Any, Literal

import numpy as np
import torch

from vision_lens.analysis.anyup import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    is_anyup_interpolation,
    prepare_anyup_image,
)
from vision_lens.analysis.patch_pca import ForegroundThreshold, RGBFitScope
from vision_lens.config import PatchPCAAnalysisConfig, VisionLensConfig
from vision_lens.models import LoadedModel


def apply_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pca_fit_settings(
    analysis: PatchPCAAnalysisConfig,
) -> tuple[bool, ForegroundThreshold, Literal["high", "low"], RGBFitScope]:
    if analysis.foreground_separation is not True:
        return False, 0.5, "high", "all"
    if (
        analysis.foreground_threshold is None
        or analysis.foreground_side is None
        or analysis.rgb_fit_scope is None
    ):
        raise ValueError("Foreground-separated patch PCA requires fit settings.")
    return (
        True,
        analysis.foreground_threshold,
        analysis.foreground_side,
        analysis.rgb_fit_scope,
    )


def anyup_guidance(
    config: VisionLensConfig,
    loaded_model: LoadedModel,
    inputs: Any,
) -> Any | None:
    if not is_anyup_interpolation(config.visualization.interpolation):
        return None
    source_mean = None
    source_std = None
    if config.preprocessing.normalize:
        source_mean = (
            config.preprocessing.mean
            or loaded_model.metadata.data_config.get("mean", IMAGENET_MEAN)
        )
        source_std = config.preprocessing.std or loaded_model.metadata.data_config.get(
            "std", IMAGENET_STD
        )
    return prepare_anyup_image(
        inputs,
        source_mean=source_mean,
        source_std=source_std,
    ).to(loaded_model.metadata.device)
