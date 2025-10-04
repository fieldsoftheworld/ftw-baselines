# Released Model Registry Dictionary

from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, Field, field_validator

RELEASE_URL = "https://github.com/fieldsoftheworld/ftw-baselines/releases/download/"

TWO_CLASS_CCBY = "2_Class_CCBY_FTW_Pretrained.ckpt"
TWO_CLASS_FULL = "2_Class_FULL_FTW_Pretrained.ckpt"
THREE_CLASS_CCBY = "3_Class_CCBY_FTW_Pretrained.ckpt"
THREE_CLASS_FULL = "3_Class_FULL_FTW_Pretrained.ckpt"


class ModelSpec(BaseModel):
    """Pydantic model with automatic validation."""

    url: str
    description: str = Field(min_length=1)
    license: Literal["CC BY 4.0", "proprietary"]
    version: str = Field(description="Model version (e.g., v1, v2, v3)")
    requires_window: bool = True
    requires_polygonize: bool = True

    @field_validator("url")
    @classmethod
    def url_must_be_ckpt(cls, v):
        if not v.endswith(".ckpt"):
            raise ValueError("URL must end with .ckpt")
        return v

    @field_validator("description")
    @classmethod
    def description_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Description cannot be empty")
        return v

    class Config:
        frozen = True  # Make immutable


MODEL_REGISTRY = {
    "2_Class_CCBY_v1": ModelSpec(
        url=f"{RELEASE_URL}v1/{TWO_CLASS_CCBY}",
        description="2-Class CCBY FTW Pretrained Model",
        license="CC BY 4.0",
        version="v1",
    ),
    "2_Class_FULL_v1": ModelSpec(
        url=f"{RELEASE_URL}v1/{TWO_CLASS_FULL}",
        description="2-Class FULL FTW Pretrained Model",
        license="CC BY 4.0",
        version="v1",
    ),
    "3_Class_CCBY_v1": ModelSpec(
        url=f"{RELEASE_URL}v1/{THREE_CLASS_CCBY}",
        description="3-Class CCBY FTW Pretrained Model",
        license="CC BY 4.0",
        version="v1",
    ),
    "3_Class_FULL_v1": ModelSpec(
        url=f"{RELEASE_URL}v1/{THREE_CLASS_FULL}",
        description="3-Class FULL FTW Pretrained Model",
        license="proprietary",
        version="v1",
    ),
    "3_Class_FULL_singleWindow_v2": ModelSpec(
        url=f"{RELEASE_URL}v2/3_Class_FULL_FTW_Pretrained_singleWindow_v2.ckpt",
        description="3-Class FULL FTW Pretrained Model (Single Window)",
        license="CC BY 4.0",
        version="v2",
    ),
    "3_Class_FULL_multiWindow_v2": ModelSpec(
        url=f"{RELEASE_URL}v2/3_Class_FULL_FTW_Pretrained_v2.ckpt",
        description="3-Class FULL FTW Pretrained Model (Multi Window)",
        license="CC BY 4.0",
        version="v2",
    ),
}
