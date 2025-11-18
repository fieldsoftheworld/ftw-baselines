import tempfile

import pytest
import torch

from ftw_tools.inference.model_registry import MODEL_REGISTRY, RELEASE_URL, ModelSpec
from ftw_tools.inference.models import load_model_from_checkpoint


@pytest.mark.integration
@pytest.mark.parametrize("model_name", MODEL_REGISTRY.keys())
def test_model_loading(model_name: str):
    if model_name.startswith("DelineateAnything"):
        return

    url = MODEL_REGISTRY[model_name].url
    with tempfile.NamedTemporaryFile(suffix=".ckpt") as tmp_file:
        torch.hub.download_url_to_file(url=url, dst=tmp_file.name)
        load_model_from_checkpoint(tmp_file.name)


def test_valid_model_spec_instance():
    model = ModelSpec(
        url=f"{RELEASE_URL}v1/2_Class_CCBY_FTW_Pretrained.ckpt",
        description="A valid model description.",
        license="CC BY 4.0",
        version="v1",
        requires_window=True,
        requires_polygonize=True,
    )
    assert model.url == RELEASE_URL + "v1/" + "2_Class_CCBY_FTW_Pretrained.ckpt"
    assert model.description == "A valid model description."
    assert model.license == "CC BY 4.0"
    assert model.version == "v1"
    assert model.requires_window is True
    assert model.requires_polygonize is True


def test_all_urls_are_valid_https():
    for model_name, spec in MODEL_REGISTRY.items():
        assert str(spec.url).startswith("https://"), (
            f"{model_name} URL must use HTTPS: {spec.url}"
        )


def test_all_urls_end_with_ckpt_or_pt():
    for model_name, spec in MODEL_REGISTRY.items():
        assert str(spec.url).endswith((".ckpt", ".pt")), (
            f"{model_name} URL must end with .ckpt or .pt: {spec.url}"
        )


def test_all_registry_values_are_model_specs():
    for model_name, value in MODEL_REGISTRY.items():
        assert isinstance(value, ModelSpec), (
            f"{model_name} value is not a ModelSpec: {type(value)}"
        )


def test_model_names_are_unique():
    keys = list(MODEL_REGISTRY.keys())
    assert len(keys) == len(set(keys)), "Duplicate model names found"
