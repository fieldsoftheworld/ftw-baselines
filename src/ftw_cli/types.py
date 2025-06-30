import enum


class ModelVersions(enum.StrEnum):
    """Mapping from short_name to .ckpt file in github."""

    TWO_CLASS_CCBY = "2_Class_CCBY_FTW_Pretrained.ckpt"
    TWO_CLASS_FULL = "2_Class_FULL_FTW_Pretrained.ckpt"
    THREE_CLASS_CCBY = "3_Class_CCBY_FTW_Pretrained.ckpt"
    THREE_CLASS_FULL = "3_Class_FULL_FTW_Pretrained.ckpt"
