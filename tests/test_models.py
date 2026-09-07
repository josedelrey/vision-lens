from types import SimpleNamespace

from vision_lens.models import _accepted_timm_image_size


def test_fixed_timm_model_uses_the_size_it_actually_accepts():
    model = SimpleNamespace(
        patch_embed=SimpleNamespace(strict_img_size=True, img_size=(224, 224))
    )

    assert _accepted_timm_image_size(model, 672) == (224, 224)


def test_dynamic_timm_model_uses_the_requested_672_size():
    model = SimpleNamespace(
        patch_embed=SimpleNamespace(strict_img_size=False, img_size=(224, 224))
    )

    assert _accepted_timm_image_size(model, 672) == (672, 672)
