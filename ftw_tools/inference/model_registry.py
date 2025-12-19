# Released Model Registry Dictionary

from typing import Literal

from pydantic import BaseModel, Field, field_validator

RELEASE_URL = "https://github.com/fieldsoftheworld/ftw-baselines/releases/download/"
HUGGINGFACE_URL = "https://hf.co/torchgeo/delineate-anything/"

TWO_CLASS_CCBY = "2_Class_CCBY_FTW_Pretrained.ckpt"
TWO_CLASS_FULL = "2_Class_FULL_FTW_Pretrained.ckpt"
THREE_CLASS_CCBY = "3_Class_CCBY_FTW_Pretrained.ckpt"
THREE_CLASS_FULL = "3_Class_FULL_FTW_Pretrained.ckpt"


class ModelSpec(BaseModel):
    """Pydantic model with automatic validation."""

    url: str
    title: str
    description: str = Field(min_length=1)
    license: Literal["CC-BY-4.0", "AGPL-3", "Mixed Open Licenses"]
    version: str = Field(description="Model version (e.g., v1, v2, v3)")
    requires_window: bool = True
    requires_polygonize: bool = True
    default: bool = False
    legacy: bool = False

    @field_validator("url")
    @classmethod
    def url_must_be_ckpt(cls, v):
        if not (v.endswith(".ckpt") or v.endswith(".pt")):
            raise ValueError("URL must end with .ckpt or .pt")
        return v

    @field_validator("description")
    @classmethod
    def description_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Description cannot be empty")
        return v


# The standard for titles is 3-Class, Full, Multi Window, B3
# If there's no difference for these within the versions,
# we don't mention them in the title to keep them short
MODEL_REGISTRY = {
    "FTW_v1_2_Class_CCBY": ModelSpec(
        title="FTW v1: CC-BY, 2-Class",
        url=f"{RELEASE_URL}v1/{TWO_CLASS_CCBY}",
        description="A two class (field / not-field) model trained on only CC-BY or CC0 input data, that was part of the first release of FTW Baseline models. It works less well than the latest models, but can be used for historical comparison, or by those who require CC-BY license before the newer options are trained on the subset of data. Requires two time windows, at the start and end of the growing season.",
        license="CC-BY-4.0",
        version="v1",
        legacy=True,
    ),
    "FTW_v1_2_Class_FULL": ModelSpec(
        title="FTW v1: Standard, 2-Class",
        url=f"{RELEASE_URL}v1/{TWO_CLASS_FULL}",
        description="A two class (field / not-field) model trained on a variety of open data licenses (including CC-BY-NC-SA and non-CC open data licenses), that was part of the first release of FTW Baseline models. It works less well than the latest models, but can be used for historical comparison. Requires two time windows, at the start and end of the growing season.",
        license="Mixed Open Licenses",
        version="v1",
        legacy=True,
    ),
    "FTW_v1_3_Class_CCBY": ModelSpec(
        title="FTW v1: CC-BY, 3-Class",
        url=f"{RELEASE_URL}v1/{THREE_CLASS_CCBY}",
        description="A three class (field, boundary, neither) model trained on only CC-BY or CC0 input data, that was part of the first release of FTW Baseline models. It works less well than the latest models, but can be used for historical comparison, or by those who require CC-BY license before the newer options are trained on the subset of data. Requires two time windows, at the start and end of the growing season.",
        license="CC-BY-4.0",
        version="v1",
        legacy=True,
    ),
    "FTW_v1_3_Class_FULL": ModelSpec(
        title="FTW v1: Standard, 3-Class",
        url=f"{RELEASE_URL}v1/{THREE_CLASS_FULL}",
        description="A three class (field, boundary, neither) model trained on a variety of open data licenses (including CC-BY-NC-SA and non-CC open data licenses), that was part of the first release of FTW Baseline models. It works less well than the latest models, but can be used for historical comparison. Requires two time windows, at the start and end of the growing season.",
        license="Mixed Open Licenses",
        version="v1",
        legacy=True,
    ),
    "FTW_v2_3_Class_FULL_singleWindow": ModelSpec(
        title="FTW v2: Standard, Single Window",
        url=f"{RELEASE_URL}v2/3_Class_FULL_FTW_Pretrained_singleWindow_v2.ckpt",
        description="A three class (field, boundary, neither) model trained on a variety of open data licenses (including CC-BY-NC-SA and non-CC open data licenses), that was part of the FTW Baseline v2 release to demonstrate using FTW data for 'one-shot' (single time window) output. Will perform a bit less well than the latest 2 window models. Requires a single time window - automatic selection will get one at the beginning of the grow season, but other times can be tried.",
        license="Mixed Open Licenses",
        version="v2",
        requires_window=False,
        legacy=True,
    ),
    "FTW_v2_3_Class_FULL_multiWindow": ModelSpec(
        title="FTW v2: Standard, Multi Window",
        url=f"{RELEASE_URL}v2/3_Class_FULL_FTW_Pretrained_v2.ckpt",
        description="A three class (field, boundary, neither) model trained on a variety of open data licenses (including CC-BY-NC-SA and non-CC open data licenses), that is the main FTW Baseline v2 release. Generally recommended for a variety of use. Requires two time windows, at the start and end of the growing season.",
        license="Mixed Open Licenses",
        version="v2",
        legacy=True,
    ),
    "DelineateAnything-S": ModelSpec(
        title="DelineateAnything v1: Mini",
        url="https://hf.co/torchgeo/delineate-anything-s/resolve/"
        "69cd440b0c5bd450ced145e68294aa9393ddae05/delineate_anything_s_rgb_yolo11n-b879d643.pt",
        description="A single-shot model trained on Field Boundary Instance Segmentation - 22M dataset (FBIS-22M). This is the small version of the model, which will run faster but with a bit less accuracy. Requires a single time window - automatic selection will get one at the beginning of the grow season, but other times can be tried.",
        license="AGPL-3",
        version="v1",
        requires_window=False,
        requires_polygonize=False,
    ),
    "DelineateAnything": ModelSpec(
        title="DelineateAnything v1: Standard",
        url="https://hf.co/torchgeo/delineate-anything/resolve/"
        "60bea7b2f81568d16d5c75e4b5b06289e1d7efaf/delineate_anything_rgb_yolo11x-88ede029.pt",
        description="A single-shot model trained on Field Boundary Instance Segmentation - 22M dataset (FBIS-22M). This is the full version of the model, which will be more accurate but will run a bit more slowly. Requires a single time window - automatic selection will get one at the beginning of the grow season, but other times can be tried.",
        license="AGPL-3",
        version="v1",
        requires_window=False,
        requires_polygonize=False,
    ),
    "FTW_PRUE_EFNET_B3": ModelSpec(
        title="FTW v3: Standard, B3",
        url=f"{RELEASE_URL}v3/prue_efnet3_checkpoint.ckpt",
        description="A three class (field, boundary, neither) model trained with EfficientNet-B3 on a variety of open data licenses (including CC-BY-NC-SA and non-CC open data licenses), that is the part of the FTW Baseline v3 release. It along with the other two EfficientNet models will likely perform best. B3 is likely the fastest of FTW v3, but less accurate than B5 and B7. Requires two time windows, at the start and end of the growing season.",
        license="Mixed Open Licenses",
        version="v3",
        requires_window=True,
        requires_polygonize=True,
    ),
    "FTW_PRUE_EFNET_B5": ModelSpec(
        title="FTW v3: Standard, B5",
        url=f"{RELEASE_URL}v3/prue_efnet5_checkpoint.ckpt",
        description="A three class (field, boundary, neither) model trained with EfficientNet-B5 on a variety of open data licenses (including CC-BY-NC-SA and non-CC open data licenses), that is the part of the FTW Baseline v3 release. It along with the other two EfficientNet models will likely perform best. B5 is a compromise of speed and accuracy in FTW v3, so slower and more accurate than B3 but faster and less accurate than B7. Requires two time windows, at the start and end of the growing season.",
        license="Mixed Open Licenses",
        version="v3",
        default=True,
    ),
    "FTW_PRUE_EFNET_B7": ModelSpec(
        title="FTW v3: Standard, B7",
        url=f"{RELEASE_URL}v3/prue_efnet7_checkpoint.ckpt",
        description="A three class (field, boundary, neither) model trained with EfficientNet-B7 on a variety of open data licenses (including CC-BY-NC-SA and non-CC open data licenses), that is the part of the FTW Baseline v3 release. It along with the other two EfficientNet models will likely perform best. B7 is likely the most accurate of FTW v3, but the slowest. Requires two time windows, at the start and end of the growing season.",
        license="Mixed Open Licenses",
        version="v3",
    ),
}
