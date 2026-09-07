from pathlib import Path

from PIL import Image

from vision_lens.processing import iter_input_batches, unique_input_labels


def test_input_batches_load_only_the_configured_batch_size(monkeypatch, tmp_path):
    from vision_lens import processing

    paths = tuple(tmp_path / f"image-{index}.jpg" for index in range(5))
    loaded_sizes = []

    def fake_load_images(batch_paths, workers):
        loaded_sizes.append((len(batch_paths), workers))
        return [Image.new("RGB", (4, 4), "white") for _ in batch_paths]

    monkeypatch.setattr(processing, "load_images", fake_load_images)

    batches = list(
        iter_input_batches(
            paths,
            unique_input_labels(paths),
            batch_size=2,
            workers=3,
        )
    )

    assert loaded_sizes == [(2, 3), (2, 3), (1, 3)]
    assert [batch.count for batch in batches] == [2, 2, 1]


def test_duplicate_filename_stems_receive_stable_unique_labels(tmp_path):
    paths = (
        Path(tmp_path / "first" / "image.jpg"),
        Path(tmp_path / "second" / "image.png"),
        Path(tmp_path / "unique.jpg"),
    )

    first = unique_input_labels(paths)
    second = unique_input_labels(paths)

    assert first == second
    assert first[0].startswith("image_")
    assert first[1].startswith("image_")
    assert first[0] != first[1]
    assert first[2] == "unique"
