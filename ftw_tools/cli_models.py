import textwrap

import click

from ftw_tools.inference.model_registry import MODEL_REGISTRY, ModelSpec


def resolve_model_name(name: str) -> str:
    lowered = name.lower()
    for model_name in MODEL_REGISTRY:
        if model_name.lower() == lowered:
            return model_name
    raise click.BadParameter(
        f"Unknown model '{name}'. Use `ftw model list` to see available models."
    )


def visible_models(include_legacy: bool) -> list[tuple[str, ModelSpec]]:
    items = list(MODEL_REGISTRY.items())
    if not include_legacy:
        items = [(name, spec) for name, spec in items if not spec.legacy]

    return sorted(
        items,
        key=lambda item: (
            not item[1].default,
            item[1].legacy,
            item[1].instance_segmentation,
            item[1].version,
            item[0].lower(),
        ),
    )


def task_label(spec: ModelSpec) -> str:
    return (
        "instance segmentation"
        if spec.instance_segmentation
        else "semantic segmentation"
    )


def inputs_label(spec: ModelSpec) -> str:
    return "1 window" if not spec.requires_window else "2 windows"


def output_label(spec: ModelSpec) -> str:
    return "direct polygons" if not spec.requires_polygonize else "raster + polygonize"


def wrap_description(text: str, indent: str = "  ") -> str:
    return textwrap.fill(
        text, width=88, initial_indent=indent, subsequent_indent=indent
    )
