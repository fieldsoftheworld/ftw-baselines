import pytest
from pydantic import ValidationError

from ftw_tools.models.model_registry import MODEL_REGISTRY, RELEASE_URL, ModelSpec


def test_valid_model_spec_instance():
    model = ModelSpec(
        url=RELEASE_URL + "v1/" + "2_Class_CCBY_FTW_Pretrained.ckpt",
        description="A valid model description.",
        license="CC BY 4.0",
        version="v1",
        requires_window=True,
        requires_polygonize=True,
    )
    assert str(model.url) == RELEASE_URL + "v1/" + "2_Class_CCBY_FTW_Pretrained.ckpt"
    assert model.description == "A valid model description."
    assert model.license == "CC BY 4.0"
    assert model.version == "v1"
    assert model.requires_window is True
    assert model.requires_polygonize is True


def test_all_urls_are_valid_https():
    """Test that all registry URLs are valid HTTPS URLs."""
    for model_name, spec in MODEL_REGISTRY.items():
        assert str(spec.url).startswith("https://"), (
            f"{model_name} URL must use HTTPS: {spec.url}"
        )


def test_all_urls_end_with_ckpt():
    """Test that all registry URLs point to .ckpt files."""
    for model_name, spec in MODEL_REGISTRY.items():
        assert str(spec.url).endswith(".ckpt"), (
            f"{model_name} URL must end with .ckpt: {spec.url}"
        )


def test_all_registry_values_are_model_specs():
    """Test that all registry values are ModelSpec instances."""
    for model_name, value in MODEL_REGISTRY.items():
        assert isinstance(value, ModelSpec), (
            f"{model_name} value is not a ModelSpec: {type(value)}"
        )


def test_model_names_are_unique():
    """Test that all model names in registry are unique."""
    keys = list(MODEL_REGISTRY.keys())
    assert len(keys) == len(set(keys)), "Duplicate model names found"
