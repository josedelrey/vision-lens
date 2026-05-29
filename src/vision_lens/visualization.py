from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

from vision_lens.attention import LayerAttentionMaps


def render_heatmap(
    attention_map: Any,
    cmap: str = "magma",
    batch_index: int = 0,
    head_index: int = 0,
) -> Any:
    array = attention_map_to_array(
        attention_map,
        batch_index=batch_index,
        head_index=head_index,
    )
    colorized = _colormap(array, cmap)
    return _image_from_array(colorized)


def overlay_attention(
    image: Any,
    attention_map: Any,
    alpha: float = 0.45,
    cmap: str = "magma",
    batch_index: int = 0,
    head_index: int = 0,
) -> Any:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")

    base_image = _as_rgb_image(image)
    heatmap = render_heatmap(
        attention_map,
        cmap=cmap,
        batch_index=batch_index,
        head_index=head_index,
    ).resize(base_image.size)

    return Image.blend(base_image, heatmap, alpha=alpha)


def make_layer_comparison_grid(
    image: Any,
    layers: Sequence[LayerAttentionMaps],
    output_path: str | Path | None = None,
    alpha: float = 0.45,
    cmap: str = "magma",
    batch_index: int = 0,
    head_index: int = 0,
    columns: int | None = None,
) -> Any:
    tiles = [
        labeled_image(
            overlay_attention(
                image,
                layer.maps,
                alpha=alpha,
                cmap=cmap,
                batch_index=batch_index,
                head_index=head_index,
            ),
            f"layer {layer.layer_index}",
        )
        for layer in layers
    ]
    grid = image_grid(tiles, columns=columns)
    if output_path is not None:
        save_image(grid, output_path)
    return grid


def make_image_comparison_grid(
    images: Sequence[Any],
    attention_maps: Sequence[Any],
    labels: Sequence[str] | None = None,
    output_path: str | Path | None = None,
    alpha: float = 0.45,
    cmap: str = "magma",
    batch_index: int = 0,
    head_index: int = 0,
    columns: int | None = None,
) -> Any:
    if len(images) != len(attention_maps):
        raise ValueError("images and attention_maps must have the same length.")

    if labels is None:
        labels = [f"image {index + 1}" for index in range(len(images))]
    if len(labels) != len(images):
        raise ValueError("labels must match the number of images.")

    tiles = [
        labeled_image(
            overlay_attention(
                image,
                attention_map,
                alpha=alpha,
                cmap=cmap,
                batch_index=batch_index,
                head_index=head_index,
            ),
            label,
        )
        for image, attention_map, label in zip(images, attention_maps, labels)
    ]
    grid = image_grid(tiles, columns=columns)
    if output_path is not None:
        save_image(grid, output_path)
    return grid


def image_grid(
    images: Sequence[Any],
    columns: int | None = None,
    background: str = "white",
    gap: int = 12,
) -> Any:
    if not images:
        raise ValueError("image_grid requires at least one image.")

    pil_images = [_as_rgb_image(image) for image in images]
    columns = _grid_columns(len(pil_images), columns)
    rows = (len(pil_images) + columns - 1) // columns
    tile_width = max(image.width for image in pil_images)
    tile_height = max(image.height for image in pil_images)

    width = columns * tile_width + (columns - 1) * gap
    height = rows * tile_height + (rows - 1) * gap
    grid = Image.new("RGB", (width, height), background)

    for index, image in enumerate(pil_images):
        row, column = divmod(index, columns)
        x = column * (tile_width + gap)
        y = row * (tile_height + gap)
        grid.paste(image, (x, y))

    return grid


def labeled_image(image: Any, label: str, label_height: int = 28) -> Any:
    base_image = _as_rgb_image(image)
    labeled = Image.new(
        "RGB",
        (base_image.width, base_image.height + label_height),
        "white",
    )
    labeled.paste(base_image, (0, label_height))

    draw = ImageDraw.Draw(labeled)
    draw.text((8, 6), label, fill="black")
    return labeled


def save_image(image: Any, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _as_rgb_image(image).save(output_path)
    return output_path


def attention_map_to_array(
    attention_map: Any,
    batch_index: int = 0,
    head_index: int = 0,
) -> Any:
    array = _to_numpy(attention_map)
    if array.ndim == 4:
        array = array[batch_index, head_index]
    elif array.ndim == 3:
        array = array[batch_index]
    elif array.ndim != 2:
        raise ValueError("attention_map must be a 2D, 3D, or 4D array/tensor.")

    array = array.astype("float32", copy=False)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum > minimum:
        return (array - minimum) / (maximum - minimum)
    return np.zeros_like(array)


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _colormap(array: Any, cmap: str) -> Any:
    colorized = matplotlib.colormaps[cmap](array)[..., :3]
    return (colorized * 255).astype(np.uint8)


def _image_from_array(array: Any) -> Any:
    return Image.fromarray(array).convert("RGB")


def _as_rgb_image(image: Any) -> Any:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(_to_numpy(image)).convert("RGB")


def _grid_columns(item_count: int, columns: int | None) -> int:
    if columns is not None:
        if columns <= 0:
            raise ValueError("columns must be positive.")
        return columns

    return min(item_count, 3)
