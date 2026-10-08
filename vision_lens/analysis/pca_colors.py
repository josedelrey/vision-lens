"""Deterministic component colors and RGB blending for higher dimensional PCA."""

from functools import lru_cache
from typing import Any

import numpy as np
import torch


def _oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB to Oklab for perceptual palette spacing."""
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    cones = (
        linear
        @ np.array(
            [
                [0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005],
            ]
        ).T
    )
    return (
        np.cbrt(cones)
        @ np.array(
            [
                [0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660],
            ]
        ).T
    )


@lru_cache(maxsize=4)
def _candidates(levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample fully saturated, maximum-value colors along six RGB cube edges."""
    grid = np.linspace(0, 1, levels)
    zero = np.zeros(levels)
    one = np.ones(levels)
    rgb = np.concatenate(
        [
            np.column_stack(channels)
            for channels in (
                (one, zero, grid),
                (one, grid, zero),
                (zero, one, grid),
                (grid, one, zero),
                (zero, grid, one),
                (grid, zero, one),
            )
        ]
    )
    rgb = np.unique(rgb, axis=0)
    return rgb, _oklab(rgb)


@lru_cache(maxsize=16)
def component_palette(dimensions: int) -> np.ndarray:
    """Start with RGB, then greedily select the most distant candidate color."""
    colors = list(np.eye(3, dtype=np.float32))
    if dimensions == 3:
        return np.eye(3, dtype=np.float32)
    levels = 17
    candidates, perceptual = _candidates(levels)
    nearest = np.full(len(candidates), np.inf)
    for color in colors:
        nearest = np.minimum(nearest, ((perceptual - _oklab(color)) ** 2).sum(1))
    while len(colors) < dimensions:
        index = int(nearest.argmax())
        if nearest[index] <= 1e-14:
            levels = 2 * levels - 1
            candidates, perceptual = _candidates(levels)
            nearest = np.full(len(candidates), np.inf)
            for color in colors:
                nearest = np.minimum(
                    nearest, ((perceptual - _oklab(color)) ** 2).sum(1)
                )
            continue
        colors.append(candidates[index])
        nearest = np.minimum(nearest, ((perceptual - perceptual[index]) ** 2).sum(1))
    palette = np.asarray(colors, dtype=np.float32)
    palette.setflags(write=False)
    return palette


def blend_component_scores(scores: Any, colors: Any) -> Any:
    """Blend normalized component scores while preserving the RGB color ratios."""
    if scores.shape[-1] == 3:
        return scores
    colors = torch.as_tensor(colors, dtype=scores.dtype, device=scores.device)
    mixed = scores @ colors
    return mixed / mixed.amax(dim=-1, keepdim=True).clamp_min(1)
