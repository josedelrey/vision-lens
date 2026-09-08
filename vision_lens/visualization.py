from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure
from PIL import Image, ImageDraw

from vision_lens.attention import LayerAttentionMaps

GRID_TILE_SIZE = (224, 224)
GRID_LABEL_HEIGHT = 32
GRID_GAP = 12
GRID_DPI = 100
GRID_LABEL_FONT_SIZE = 10
RASTER_GRID_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".raw",
    ".rgba",
    ".tif",
    ".tiff",
    ".webp",
}


def render_heatmap(
    attention_map: Any,
    cmap: str = "viridis",
    batch_index: int = 0,
    head_index: int = 0,
    normalization: str = "per_map",
    normalization_range: tuple[float, float] | None = None,
) -> Any:
    array = attention_map_to_array(
        attention_map,
        batch_index=batch_index,
        head_index=head_index,
        normalization=normalization,
        normalization_range=normalization_range,
    )
    colorized = _colormap(array, cmap)
    return _image_from_array(colorized)


def overlay_attention(
    image: Any,
    attention_map: Any,
    alpha: float = 0.45,
    cmap: str = "viridis",
    batch_index: int = 0,
    head_index: int = 0,
    normalization: str = "per_map",
    normalization_range: tuple[float, float] | None = None,
) -> Any:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")

    base_image = _as_rgb_image(image)
    heatmap = render_heatmap(
        attention_map,
        cmap=cmap,
        batch_index=batch_index,
        head_index=head_index,
        normalization=normalization,
        normalization_range=normalization_range,
    ).resize(base_image.size)

    return Image.blend(base_image, heatmap, alpha=alpha)


def make_layer_comparison_grid(
    image: Any,
    layers: Sequence[LayerAttentionMaps],
    output_path: str | Path | None = None,
    alpha: float = 0.45,
    cmap: str = "viridis",
    batch_index: int = 0,
    head_index: int = 0,
    columns: int | None = None,
    tile_size: tuple[int, int] | None = None,
    spacing: int | None = None,
    padding: int | None = None,
    show_labels: bool | None = None,
    background: str | None = None,
    dpi: int | None = None,
    normalization: str = "per_map",
    normalization_range: tuple[float, float] | None = None,
) -> Any:
    labels = [f"layer {layer.layer_index}" for layer in layers]
    overlays = [
        overlay_attention(
            image,
            layer.maps,
            alpha=alpha,
            cmap=cmap,
            batch_index=batch_index,
            head_index=head_index,
            normalization=normalization,
            normalization_range=normalization_range,
        )
        for layer in layers
    ]
    resolved_labels = True if show_labels is None else show_labels
    tiles = [
        labeled_image(overlay, label) if resolved_labels else overlay
        for overlay, label in zip(overlays, labels, strict=True)
    ]
    grid = image_grid(
        tiles,
        columns=columns,
        background=background or "white",
        gap=12 if spacing is None else spacing,
        padding=0 if padding is None else padding,
        tile_size=tile_size,
    )
    if output_path is not None:
        save_grid(
            overlays,
            labels,
            output_path,
            columns=columns,
            tile_size=tile_size,
            spacing=spacing,
            padding=padding,
            show_labels=resolved_labels,
            background=background,
            dpi=dpi,
        )
    return grid


def make_image_comparison_grid(
    images: Sequence[Any],
    attention_maps: Sequence[Any],
    labels: Sequence[str] | None = None,
    output_path: str | Path | None = None,
    alpha: float = 0.45,
    cmap: str = "viridis",
    batch_index: int = 0,
    head_index: int = 0,
    columns: int | None = None,
    tile_size: tuple[int, int] | None = None,
    spacing: int | None = None,
    padding: int | None = None,
    show_labels: bool | None = None,
    background: str | None = None,
    dpi: int | None = None,
    normalization: str = "per_map",
    normalization_range: tuple[float, float] | None = None,
) -> Any:
    if len(images) != len(attention_maps):
        raise ValueError("images and attention_maps must have the same length.")

    if labels is None:
        labels = [f"image {index + 1}" for index in range(len(images))]
    if len(labels) != len(images):
        raise ValueError("labels must match the number of images.")

    overlays = [
        overlay_attention(
            image,
            attention_map,
            alpha=alpha,
            cmap=cmap,
            batch_index=batch_index,
            head_index=head_index,
            normalization=normalization,
            normalization_range=normalization_range,
        )
        for image, attention_map in zip(images, attention_maps, strict=True)
    ]
    resolved_labels = True if show_labels is None else show_labels
    tiles = [
        labeled_image(overlay, label) if resolved_labels else overlay
        for overlay, label in zip(overlays, labels, strict=True)
    ]
    grid = image_grid(
        tiles,
        columns=columns,
        background=background or "white",
        gap=12 if spacing is None else spacing,
        padding=0 if padding is None else padding,
        tile_size=tile_size,
    )
    if output_path is not None:
        save_grid(
            overlays,
            labels,
            output_path,
            columns=columns,
            tile_size=tile_size,
            spacing=spacing,
            padding=padding,
            show_labels=resolved_labels,
            background=background,
            dpi=dpi,
        )
    return grid


def save_grid(
    images: Sequence[Any],
    labels: Sequence[str],
    output_path: str | Path,
    columns: int | None = None,
    tile_size: tuple[int, int] | None = None,
    spacing: int | None = None,
    padding: int | None = None,
    show_labels: bool = True,
    background: str | None = None,
    dpi: int | None = None,
) -> Path:
    return save_grid_figure(
        images,
        labels=labels,
        output_path=output_path,
        columns=columns,
        tile_size=tile_size,
        spacing=spacing,
        padding=padding,
        show_labels=show_labels,
        background=background,
        dpi=dpi,
    )


def save_grid_figure(
    images: Sequence[Any],
    labels: Sequence[str],
    output_path: str | Path,
    columns: int | None = None,
    tile_size: tuple[int, int] | None = None,
    spacing: int | None = None,
    padding: int | None = None,
    show_labels: bool = True,
    background: str | None = None,
    dpi: int | None = None,
) -> Path:
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length.")
    if not images:
        raise ValueError("save_grid_figure requires at least one image.")

    resolved_tile_size = tile_size or GRID_TILE_SIZE
    pil_images = [
        _fit_image_to_tile(_as_rgb_image(image), resolved_tile_size) for image in images
    ]
    columns = _grid_columns(len(pil_images), columns)
    rows = (len(pil_images) + columns - 1) // columns
    tile_width, tile_height = resolved_tile_size
    label_height = GRID_LABEL_HEIGHT if show_labels else 0
    gap = GRID_GAP if spacing is None else spacing
    margin = 0 if padding is None else padding
    resolved_dpi = GRID_DPI if dpi is None else dpi
    width = columns * tile_width + (columns - 1) * gap + 2 * margin
    height = rows * (tile_height + label_height) + (rows - 1) * gap + 2 * margin
    output = Path(output_path)
    pixel_bias = 1e-6 if output.suffix.lower() in RASTER_GRID_FORMATS else 0
    figure = Figure(
        figsize=(
            (width + pixel_bias) / resolved_dpi,
            (height + pixel_bias) / resolved_dpi,
        ),
        dpi=resolved_dpi,
        frameon=False,
    )
    if background is not None:
        figure.patch.set_facecolor(background)

    for index, (image, label) in enumerate(zip(pil_images, labels, strict=True)):
        row, column = divmod(index, columns)
        cell_x = margin + column * (tile_width + gap)
        cell_y = margin + row * (tile_height + label_height + gap)
        image_x = cell_x
        image_y_top = cell_y + label_height
        image_y = height - image_y_top - image.height
        figure.figimage(np.asarray(image), xo=image_x, yo=image_y)
        if show_labels:
            figure.text(
                (cell_x + tile_width / 2) / width,
                1 - (cell_y + label_height / 2) / height,
                label,
                ha="center",
                va="center",
                fontsize=GRID_LABEL_FONT_SIZE,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    return output


def image_grid(
    images: Sequence[Any],
    columns: int | None = None,
    background: str = "white",
    gap: int = 12,
    padding: int = 0,
    tile_size: tuple[int, int] | None = None,
) -> Any:
    if not images:
        raise ValueError("image_grid requires at least one image.")

    pil_images = [_as_rgb_image(image) for image in images]
    if tile_size is not None:
        pil_images = [_fit_image_to_tile(image, tile_size) for image in pil_images]
    columns = _grid_columns(len(pil_images), columns)
    rows = (len(pil_images) + columns - 1) // columns
    tile_width = max(image.width for image in pil_images)
    tile_height = max(image.height for image in pil_images)

    if gap < 0 or padding < 0:
        raise ValueError("gap and padding must be non-negative.")

    width = columns * tile_width + (columns - 1) * gap + 2 * padding
    height = rows * tile_height + (rows - 1) * gap + 2 * padding
    grid = Image.new("RGB", (width, height), background)

    for index, image in enumerate(pil_images):
        row, column = divmod(index, columns)
        x = padding + column * (tile_width + gap)
        y = padding + row * (tile_height + gap)
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


def save_image(image: Any, path: str | Path, dpi: int | None = None) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    options = {} if dpi is None else {"dpi": (dpi, dpi)}
    _as_rgb_image(image).save(output_path, **options)
    return output_path


def attention_map_to_array(
    attention_map: Any,
    batch_index: int = 0,
    head_index: int = 0,
    normalization: str = "per_map",
    normalization_range: tuple[float, float] | None = None,
) -> Any:
    array = _to_numpy(attention_map)
    if array.ndim == 4:
        array = array[batch_index, head_index]
    elif array.ndim == 3:
        array = array[batch_index]
    elif array.ndim != 2:
        raise ValueError("attention_map must be a 2D, 3D, or 4D array/tensor.")

    array = array.astype("float32", copy=False)
    if normalization == "per_map":
        minimum = float(np.min(array))
        maximum = float(np.max(array))
    elif normalization in {"shared", "fixed"}:
        if normalization_range is None:
            raise ValueError(
                f"normalization_range is required for normalization={normalization!r}."
            )
        minimum, maximum = normalization_range
    else:
        raise ValueError("normalization must be one of: per_map, shared, fixed.")
    if maximum > minimum:
        return np.clip((array - minimum) / (maximum - minimum), 0, 1)
    return np.zeros_like(array)


def shared_value_range(values: Sequence[Any]) -> tuple[float, float]:
    if not values:
        raise ValueError("shared normalization requires at least one array.")
    arrays = [_to_numpy(value).astype("float32", copy=False) for value in values]
    return (
        min(float(np.min(array)) for array in arrays),
        max(float(np.max(array)) for array in arrays),
    )


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _colormap(array: Any, cmap: str) -> Any:
    colorized = matplotlib.colormaps[cmap](array)[..., :3]
    return (colorized * 255).astype(np.uint8)


def _image_from_array(array: Any) -> Any:
    return Image.fromarray(array).convert("RGB")


def _fit_image_to_tile(image: Any, tile_size: tuple[int, int]) -> Any:
    base_image = _as_rgb_image(image)
    fitted = base_image.copy()
    fitted.thumbnail(tile_size, Image.Resampling.LANCZOS)

    tile = Image.new("RGB", tile_size, "white")
    x = (tile_size[0] - fitted.width) // 2
    y = (tile_size[1] - fitted.height) // 2
    tile.paste(fitted, (x, y))
    return tile


def _as_rgb_image(image: Any) -> Any:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(_to_numpy(image)).convert("RGB")


def _grid_columns(item_count: int, columns: int | None) -> int:
    if columns is not None:
        if columns <= 0:
            raise ValueError("columns must be positive.")
        return min(columns, item_count)

    return min(item_count, 3)
