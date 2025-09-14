#!/usr/bin/env nu

# Populate pyproject.toml dependencies from pixi.toml before building

# Format a package dependency with optional version constraint
def format-dependency [key: string, value: string] {
    if $value == "*" { $key } else { $"($key)($value)" }
}

# Extract conda dependencies, excluding system packages
def extract-conda-deps [config: record] {
    $config.dependencies
        | items {|key, value|
            # Exclude system packages and packages that need PyPI name mapping
            if $key in ["python", "cuda", "libgdal-arrow-parquet", "gdal", "pip"] {
                null
            } else if $key == "pytorch" {
                # Map pytorch (conda name) to torch (pypi name)
                let version_spec = match $value {
                    "*" => ""
                    _ => $value
                }
                if $version_spec == "" {
                    "torch"
                } else {
                    $"torch($version_spec)"
                }
            } else {
                let version_spec = match $value {
                    "*" => ""
                    _ => $value
                }

                if $version_spec == "" {
                    $key
                } else {
                    $"($key)($version_spec)"
                }
            }
        }
        | where $it != null
        | sort
}

# Extract PyPI dependencies from pixi configuration
def extract-pypi-deps [config: record] {
    $config.pypi-dependencies
        | items {|key, value|
            # Skip self reference to avoid circular dependency
            if $key == "ftw-tools" {
                null
            } else if $key == "dask" {
                # Handle dask[distributed] format for pypi
                if $value == "*" {
                    "dask[distributed]"
                } else {
                    $"dask[distributed]($value)"
                }
            } else if $key == "distributed" {
                # Skip distributed as it's included in dask[distributed]
                null
            } else {
                format-dependency $key $value
            }
        }
        | where $it != null
        | sort
}

# Extract dependencies from a pixi feature (conda + pypi)
def extract-feature-deps [feature_config: record] {
    let conda_deps = if ("dependencies" in $feature_config) {
        $feature_config.dependencies
            | items {|key, value| format-dependency $key $value }
    } else { [] }
    
    let pypi_deps = if ("pypi-dependencies" in $feature_config) {
        $feature_config.pypi-dependencies
            | items {|key, value| format-dependency $key $value }
    } else { [] }
    
    $conda_deps | append $pypi_deps | sort
}

# Build optional dependencies structure from pixi features
def extract-optional-deps [config: record] {
    let features = if ("feature" in $config) { $config.feature } else { {} }
    
    $features
        | items {|feature_name, feature_config|
            {
                name: $feature_name,
                deps: (extract-feature-deps $feature_config)
            }
        }
        | reduce -f {} {|item, acc|
            $acc | upsert $item.name $item.deps
        }
        | upsert "all" [($"ftw-tools[" + ($features | items {|name, _| $name } | str join ",") + "]")]
}

def main [] {
    let pixi_config = open pixi.toml
    let pyproject = open pyproject.toml
    
    let all_deps = [
        (extract-conda-deps $pixi_config),
        (extract-pypi-deps $pixi_config)
    ] | flatten | sort
    
    let optional_deps = extract-optional-deps $pixi_config
    
    $pyproject
        | upsert project.dependencies $all_deps
        | upsert project.optional-dependencies $optional_deps
        | to toml
        | save --force pyproject.toml
}