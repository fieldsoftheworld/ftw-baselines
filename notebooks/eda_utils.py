"""Shared helpers for the FTW-Austria EDA notebook."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from geopy.distance import geodesic
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pyproj import Geod
from skimage.color import label2rgb
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

SPLIT_COLORS = {'train': '#2ca25f', 'val': '#1f78b4', 'test': '#de2d26', 'unknown': '#636363'}
WINDOW_COLORS = {'window_a': '#e6550d', 'window_b': '#3182bd'}
ESRI_WORLD_IMAGERY = (
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
)


def list_stems(folder: Path, suffix: str = '.tif', strip_prefix: str = '') -> set[str]:
    if not folder.exists():
        return set()
    stems = {p.stem for p in folder.glob(f'*{suffix}')}
    return {s.removeprefix(strip_prefix) for s in stems} if strip_prefix else stems


def common_chip_ids(df_chips: pd.DataFrame, folders: Mapping[str, Path]) -> set[str]:
    """Intersect parquet aoi_ids with stems present in every folder."""
    common = set(df_chips['aoi_id'])
    for folder in folders.values():
        common &= list_stems(folder)
    return common


def read_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        img = src.read()
    if img.shape[0] >= 3:
        rgb = np.moveaxis(img[:3], 0, -1).astype(np.float32)
    else:
        rgb = np.stack([img[0].astype(np.float32)] * 3, axis=-1)
    low, high = np.percentile(rgb, [2, 98])
    if high > low:
        return np.clip((rgb - low) / (high - low), 0, 1)
    return np.zeros_like(rgb)


def read_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def count_fields(mask_path: Path) -> int:
    ids = np.unique(read_mask(mask_path))
    return int((ids != 0).sum())


def to_lon_lat_pairs(geom_coords) -> list[list[float]]:
    """Normalize FTW geometry coordinates to a list of [lon, lat] pairs."""
    if not geom_coords:
        return []
    first = geom_coords[0]
    if isinstance(first, (list, tuple)) and len(first) == 2 and all(
        isinstance(v, (int, float)) for v in first
    ):
        return geom_coords
    if isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple)):
        second = first[0]
        if len(second) == 2 and all(isinstance(v, (int, float)) for v in second):
            return first
        if isinstance(second, (list, tuple)) and second and isinstance(second[0], (list, tuple)):
            return second
    raise ValueError('Unsupported geometry coordinate format in grid config.')


def grid_metrics_km(ring_lon_lat, geod: Geod) -> tuple[float, float, float]:
    """Return (length_km, width_km, area_km2) for one grid ring."""
    if not ring_lon_lat:
        return 0.0, 0.0, 0.0
    lons = [pt[0] for pt in ring_lon_lat]
    lats = [pt[1] for pt in ring_lon_lat]
    if ring_lon_lat[0] != ring_lon_lat[-1]:
        lons.append(ring_lon_lat[0][0])
        lats.append(ring_lon_lat[0][1])
    center_lon = 0.5 * (min(lons) + max(lons))
    center_lat = 0.5 * (min(lats) + max(lats))
    length_km = geodesic((min(lats), center_lon), (max(lats), center_lon)).kilometers
    width_km = geodesic((center_lat, min(lons)), (center_lat, max(lons))).kilometers
    area_m2, _ = geod.polygon_area_perimeter(lons, lats)
    return length_km, width_km, abs(area_m2) / 1_000_000.0


def build_grid_chips_map(
    data_config: dict,
    df_chips: pd.DataFrame,
    output_path: Path | None = None,
) -> folium.Map:
    """Render the grids+chips folium map and optionally save it."""
    geod = Geod(ellps='WGS84')
    rings: list[list[tuple[float, float]]] = []
    metrics: list[tuple[float, float, float]] = []
    all_lons: list[float] = []
    all_lats: list[float] = []

    for grid in data_config.get('grids', []):
        ring = to_lon_lat_pairs(grid['geometry']['coordinates'])
        if not ring:
            continue
        lons = [pt[0] for pt in ring]
        lats = [pt[1] for pt in ring]
        if ring[0] != ring[-1]:
            lons.append(ring[0][0])
            lats.append(ring[0][1])
        all_lons.extend(lons)
        all_lats.extend(lats)
        rings.append(list(zip(lats, lons)))
        metrics.append(grid_metrics_km(ring, geod))

    m = folium.Map(
        location=[float(np.mean(all_lats)), float(np.mean(all_lons))],
        zoom_start=7, tiles=None, control_scale=True, prefer_canvas=True,
        max_zoom=20, min_zoom=2, zoom_control=True,
        attr='Esri, Maxar, Earthstar Geographics',
    )
    folium.TileLayer(
        tiles=ESRI_WORLD_IMAGERY, name='Satellite Basemap',
        attr='Esri, Maxar, Earthstar Geographics', overlay=False, control=True,
    ).add_to(m)

    grids_layer = folium.FeatureGroup(name='Grids', show=True)
    for i, (ring, (length_km, width_km, area_km2)) in enumerate(zip(rings, metrics), start=1):
        popup = folium.Popup(
            f'<b>Grid {i}</b><br>Length: {length_km:.3f} km<br>'
            f'Width: {width_km:.3f} km<br>Area: {area_km2:.3f} km^2',
            max_width=250,
        )
        folium.Polygon(locations=ring, color='yellow', weight=2, fill=False,
                       opacity=0.9, popup=popup).add_to(grids_layer)
    grids_layer.add_to(m)

    # Render all chips as a single GeoJson layer (much smaller HTML than 6k+ folium.Polygons).
    chip_geoms = gpd.GeoSeries.from_wkb(df_chips['geometry'], crs='EPSG:4326')
    gdf = gpd.GeoDataFrame(
        {'aoi_id': df_chips['aoi_id'].astype(str), 'split': df_chips['split'].fillna('unknown').astype(str)},
        geometry=chip_geoms, crs='EPSG:4326',
    ).dropna(subset=['geometry'])
    folium.GeoJson(
        gdf,
        name='Chips',
        style_function=lambda feat: {
            'color': SPLIT_COLORS.get(feat['properties']['split'], SPLIT_COLORS['unknown']),
            'weight': 1,
            'fillOpacity': 0.15,
        },
        tooltip=folium.GeoJsonTooltip(fields=['aoi_id', 'split']),
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    legend_items = ''.join(
        f'<div><span style="display:inline-block;width:12px;height:12px;'
        f'background:{color};border:1px solid #333;margin-right:6px;"></span>{name}</div>'
        for name, color in SPLIT_COLORS.items()
    )
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:24px;right:24px;z-index:9999;background:white;'
        f'border:2px solid #333;border-radius:6px;padding:10px 12px;font-size:13px;'
        f'line-height:1.4;box-shadow:0 2px 8px rgba(0,0,0,0.25);">'
        f'<div style="font-weight:700;margin-bottom:6px;">Chip Split Legend</div>'
        f'{legend_items}</div>'
    ))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))
    return m


def plot_chip_samples(chip_ids: list[str], folders: Mapping[str, Path]) -> Figure:
    """Plot one row per chip. `folders` must have keys:
    'window_a', 'window_b', 'instance', 'semantic_2class', 'semantic_3class'.
    """
    cols = ['window_a', 'window_b', 'instance', 'semantic_2class', 'semantic_3class']
    fig, axes = plt.subplots(len(chip_ids), 5, figsize=(20, 3.8 * len(chip_ids)), squeeze=False)
    for c, title in enumerate(cols):
        axes[0, c].set_title(title, fontsize=11, fontweight='bold')
    for r, chip_id in enumerate(chip_ids):
        axes[r, 0].imshow(read_rgb(folders['window_a'] / f'{chip_id}.tif'))
        axes[r, 1].imshow(read_rgb(folders['window_b'] / f'{chip_id}.tif'))
        axes[r, 2].imshow(label2rgb(
            read_mask(folders['instance'] / f'{chip_id}.tif'),
            bg_label=0, bg_color=(0, 0, 0),
        ))
        axes[r, 3].imshow(read_mask(folders['semantic_2class'] / f'{chip_id}.tif'),
                          cmap='viridis', interpolation='nearest')
        axes[r, 4].imshow(read_mask(folders['semantic_3class'] / f'{chip_id}.tif'),
                          cmap='tab10', interpolation='nearest')
        for c in range(5):
            axes[r, c].axis('off')
        axes[r, 0].set_ylabel(chip_id, rotation=0, labelpad=45, va='center', fontsize=9)
    fig.tight_layout()
    return fig


def report_dataset_stats(
    data_config: dict,
    df_chips: pd.DataFrame,
    folders: Mapping[str, Path],
    common_ids: Iterable[str] | None = None,
) -> None:
    """Print dataset summary and show the field-size histogram."""
    instance_dir = folders['instance']
    ids = set(common_ids) if common_ids is not None else common_chip_ids(df_chips, folders)

    df_complete = df_chips[df_chips['aoi_id'].isin(ids)]
    split_counts = (
        df_complete['split'].value_counts().reindex(['train', 'val', 'test']).fillna(0).astype(int)
    )

    field_sizes: list[int] = []
    total_field_px = 0
    total_px = 0
    for chip_id in sorted(ids):
        inst = read_mask(instance_dir / f'{chip_id}.tif')
        unique_ids, counts = np.unique(inst, return_counts=True)
        sizes = counts[unique_ids != 0].astype(np.int64)
        field_sizes.extend(sizes.tolist())
        total_field_px += int(sizes.sum())
        total_px += int(inst.size)

    n_instances = len(field_sizes)
    avg = float(np.mean(field_sizes)) if n_instances else 0.0
    median = float(np.median(field_sizes)) if n_instances else 0.0
    ratio = (total_field_px / total_px) if total_px else 0.0

    print('=' * 72)
    print('FTW Austria Dataset Summary')
    print('=' * 72)
    print(f'Grids in config                    : {len(data_config.get("grids", [])):,}')
    print(f'Total chips in parquet             : {len(df_chips):,}')
    print(f'Complete chips (all modalities)    : {len(ids):,}')
    print()
    print('Split counts (complete chips):')
    for name in ('train', 'val', 'test'):
        print(f'  - {name:<5}: {int(split_counts.get(name, 0)):,}')
    print()
    print(f'Total field instances in labels    : {n_instances:,}')
    print(f'Average field size                 : {avg:,.2f} pixels')
    print(f'Median field size                  : {median:,.2f} pixels')
    print(f'Max field size                     : {max(field_sizes) if field_sizes else 0:,} pixels')
    print(f'Min field size                     : {min(field_sizes) if field_sizes else 0:,} pixels')
    print(f'Field-to-total area ratio          : {ratio:.4f} ({ratio * 100:.2f}%)')
    if ids:
        print(f'Average of fields per chip         : {n_instances / len(ids):.2f}*')
    print('=' * 72)
    print('*: This makes a dense object segmentation task.')

    if not field_sizes:
        return
    arr = np.asarray(field_sizes, dtype=np.int64)
    lo, hi = np.percentile(arr, [0.1, 99.9])
    n_small = int((arr < lo).sum())
    n_large = int((arr > hi).sum())
    print(f'Removing outliers outside [{lo:.2f}, {hi:.2f}] pixels for histogram.')
    print(f'{n_small} small and {n_large} large fields removed.')
    trimmed = arr[(arr >= lo) & (arr <= hi)]

    plt.figure(figsize=(10, 5))
    plt.hist(trimmed, bins=50, color=SPLIT_COLORS['train'], edgecolor='white', alpha=0.9)
    plt.xscale('log')
    plt.xlabel('Field size (pixels, log scale)')
    plt.ylabel('Number of fields')
    plt.title('Distribution of Field Sizes (Instance Masks)')
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.show()


def collect_gfm_features(
    aoi_ids: Iterable[str],
    window_dirs: Mapping[str, Path],
    instance_dir: Path,
    df_chips: pd.DataFrame,
    prefix: str = 'clay_',
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load mean-pooled GFM embeddings per chip and concatenate across windows.
    Returns (X, attrs) where attrs has keys:
      'split', 'grid_name', 'n_fields', 'mean_field_size', 'total_field_size', 'has_field'.
    Embeddings on disk are float16; means accumulate in float32.
    """
    aoi_ids = list(aoi_ids)
    if not aoi_ids:
        empty = np.array([], dtype=np.float32)
        return np.empty((0, 0), np.float32), {
            'split': np.array([], dtype=object),
            'grid_name': np.array([], dtype=object),
            'n_fields': empty.astype(np.int64),
            'mean_field_size': empty,
            'total_field_size': empty,
            'has_field': empty.astype(bool),
        }

    split_lookup = df_chips.set_index('aoi_id')['split']
    n = len(aoi_ids)
    splits = np.empty(n, dtype=object)
    grid_names = np.empty(n, dtype=object)
    n_fields = np.zeros(n, dtype=np.int64)
    mean_field_size = np.zeros(n, dtype=np.float32)
    total_field_size = np.zeros(n, dtype=np.float32)

    first = aoi_ids[0]
    per_window_dim = np.load(window_dirs[next(iter(window_dirs))] / f'{prefix}{first}.npz')['embedding'].shape[1]
    X = np.empty((n, per_window_dim * len(window_dirs)), dtype=np.float32)

    for i, aoi_id in enumerate(tqdm(aoi_ids, desc='Loading GFM features')):
        for j, win in enumerate(window_dirs):
            arr = np.load(window_dirs[win] / f'{prefix}{aoi_id}.npz')['embedding']
            X[i, j * per_window_dim:(j + 1) * per_window_dim] = arr.mean(axis=0, dtype=np.float32)
        splits[i] = split_lookup.get(aoi_id, 'unknown')
        grid_names[i] = aoi_id.split('_')[0]
        inst_path = instance_dir / f'{aoi_id}.tif'
        if inst_path.exists():
            inst = read_mask(inst_path)
            unique_ids, counts = np.unique(inst, return_counts=True)
            field_counts = counts[unique_ids != 0]
            n_fields[i] = len(field_counts)
            if len(field_counts):
                mean_field_size[i] = float(field_counts.mean())
                total_field_size[i] = float(field_counts.sum())

    return X, {
        'split': splits,
        'grid_name': grid_names,
        'n_fields': n_fields,
        'mean_field_size': mean_field_size,
        'total_field_size': total_field_size,
        'has_field': n_fields > 0,
    }


def fit_pca(X: np.ndarray, n_components: int = 2) -> tuple[PCA, np.ndarray]:
    """Fit PCA and return (model, transformed coords)."""
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)
    return pca, Z


def feat_pca_plot(
    Z: np.ndarray,
    evr: np.ndarray,
    attr: np.ndarray,
    *,
    attr_name: str = 'attribute',
    palette: Mapping | None = None,
    cmap: str = 'viridis',
    vmax_percentile: float | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    s: int = 10,
    alpha: float = 0.5,
) -> Axes:
    """Scatter of 2D PCA coords colored by `attr`.
    Categorical when `palette` is given or `attr` is non-numeric; otherwise continuous.
    Use `fit_pca` to get `Z` and `evr=pca.explained_variance_ratio_`.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    is_categorical = palette is not None or attr.dtype.kind in {'O', 'U', 'S', 'b'}
    if is_categorical:
        levels = list(palette.keys()) if palette else list(pd.unique(attr))
        for lvl in levels:
            mask = attr == lvl
            if not mask.any():
                continue
            ax.scatter(Z[mask, 0], Z[mask, 1], s=s, alpha=alpha,
                       color=(palette or {}).get(lvl),
                       label=f'{lvl} (n={int(mask.sum())})')
        ax.legend(fontsize=10)
    else:
        vmax = float(np.percentile(attr, vmax_percentile)) if vmax_percentile else None
        sc = ax.scatter(Z[:, 0], Z[:, 1], s=s, alpha=alpha, c=attr, cmap=cmap,
                        vmin=0 if vmax is not None else None, vmax=vmax)
        cbar = plt.colorbar(sc, ax=ax)
        label = attr_name
        if vmax is not None:
            label += f' (clipped at p{vmax_percentile:g} = {vmax:.0f})'
        cbar.set_label(label)

    ax.set_xlabel(f'PC1 ({evr[0]:.1%})')
    ax.set_ylabel(f'PC2 ({evr[1]:.1%})')
    ax.set_title(title or f'PCA of features - colored by {attr_name}')
    ax.grid(alpha=0.2)
    return ax
