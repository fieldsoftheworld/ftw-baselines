# FTW Baseline — Training Workflow

End-to-end pipeline for the 3-class U-Net + EfficientNet-B3 baseline:
**training → validation → testing → inference → post-processing → result display**,
with WandB tracking throughout.

---

## 0. Prerequisites

```bash
# Install dependencies (wandb is now included)
uv sync --all-extras --dev

# Authenticate WandB (run once)
wandb login

# Download Austria data
ftw data download --countries=Austria
```

---

## 1. Training

```bash
ftw model fit --config configs/dwei/3_class/full-ftw.yaml
```

- Config: `configs/dwei/3_class/full-ftw.yaml`
- Model: U-Net + EfficientNet-B3, 8-channel Sentinel-2 input, 3 classes
- Loss: cross-entropy with class weights `[0.04, 0.08, 0.88]`
- Checkpoint saved to: `logs/FTW-Release-Full-3-class/lightning_logs/version_0/checkpoints/last.ckpt`
- WandB project: `ftw-baselines`, run name: `3class-unet-efficientnetb3-austria`

**Validation** runs automatically after each epoch. Metrics logged to WandB:
- `val/loss`, `val/iou_macro`, `val/precision_macro`, `val/recall_macro`
- `val/object_precision`, `val/object_recall`, `val/object_f1`
- `val/predictions` — RGB / GT / Prediction image panels (4 samples per epoch)

Resume from checkpoint:
```bash
ftw model fit --config configs/dwei/3_class/full-ftw.yaml \
  --ckpt_path logs/FTW-Release-Full-3-class/.../last.ckpt
```

---

## 2. Hyperparameter Sweep (WandB)

```bash
# Create sweep (run once)
wandb sweep configs/dwei/wandb_sweep.yaml

# Launch agent for each trial
wandb agent <entity>/ftw-baselines/<sweep-id>
```

Sweep config: `configs/dwei/wandb_sweep.yaml`
Wrapper script: `scripts/ftw_model_fit.py`
Swept parameters: `lr` (log-uniform 1e-4..1e-2), `loss` (ce/focal/jaccard), `backbone` (efficientnet-b3/resnet50)

---

## 3. Model Testing & Metric Evaluation

```bash
mkdir -p outputs

ftw model test \
  --model logs/FTW-Release-Full-3-class/lightning_logs/version_0/checkpoints/last.ckpt \
  --countries austria \
  --model_predicts_3_classes \
  --test_on_3_classes \
  --bootstrap \
  --out outputs/test_metrics.csv
```

Output CSV columns:
| Column | Description |
|--------|-------------|
| `pixel_level_iou` | Jaccard index for the crop class |
| `pixel_level_precision` | Precision for the crop class |
| `pixel_level_recall` | Recall for the crop class |
| `object_level_precision` | Object-level precision (IoU ≥ 0.5) |
| `object_level_recall` | Object-level recall |
| `object_level_f1` | Object-level F1 |
| `*_ci_lower/upper` | 95% bootstrap confidence intervals |

---

## 4. Inference on a Full Scene

```bash
# Pick any Austria test TIF, or download a Sentinel-2 scene
ftw inference run \
  --model logs/FTW-Release-Full-3-class/lightning_logs/version_0/checkpoints/last.ckpt \
  --input <path-to-sentinel2.tif> \
  --output outputs/austria_pred.tif
```

The input TIF must be an 8-band Sentinel-2 stack (4 bands from each of two time windows).

---

## 5. Post-Processing (Polygonization)

```bash
ftw inference polygonize outputs/austria_pred.tif \
  --output outputs/austria_fields.parquet \
  --simplify 15 \
  --min_size 500
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--simplify` | 15 | Simplification tolerance in CRS units (meters for UTM) |
| `--min_size` | 500 | Minimum field area in m² |
| `--output` | — | Output path (`.parquet`, `.gpkg`, `.geojson`, or `.shp`) |

Output: `outputs/austria_fields.parquet` — fiboa-compatible GeoParquet

---

## 6. Result Display

### Metrics table

```python
import pandas as pd
pd.read_csv("outputs/test_metrics.csv").T
```

### Visual overlays (Jupyter notebook)

```bash
jupyter notebook notebooks/visualize_results.ipynb
```

The notebook shows:
1. **RGB** composite (Sentinel-2 bands 0,1,2 — window A)
2. **Ground truth** mask (background / crop / boundary)
3. **Predicted** mask
4. **Prediction overlay** (crop/boundary as transparent mask on RGB)
5. **Full-scene inference** raster + vectorized polygon boundaries
6. **Field size distribution** histogram

Outputs saved to `outputs/sample_predictions.png` and `outputs/full_scene_results.png`.

---

## File Map

| Path | Purpose |
|------|---------|
| `configs/dwei/3_class/full-ftw.yaml` | Training config (model, data, WandB logger) |
| `configs/dwei/wandb_sweep.yaml` | WandB hyperparameter sweep definition |
| `scripts/ftw_model_fit.py` | Sweep agent entry point |
| `ftw_tools/training/trainers.py` | Trainer with WandB image logging in `validation_step` |
| `notebooks/visualize_results.ipynb` | Result visualization notebook |
| `outputs/test_metrics.csv` | Test metrics (generated at runtime) |
| `outputs/austria_pred.tif` | Full-scene prediction raster (generated at runtime) |
| `outputs/austria_fields.parquet` | Vectorized field polygons (generated at runtime) |

---

## Verification Checklist

- [ ] `ftw model fit` completes without errors; `last.ckpt` appears in `logs/`
- [ ] WandB run appears at wandb.ai under project `ftw-baselines`
- [ ] `val/predictions` media panel shows RGB/GT/Pred images in WandB
- [ ] `ftw model test` writes `outputs/test_metrics.csv` with non-zero metrics
- [ ] `outputs/austria_pred.tif` is a valid single-band raster
- [ ] `geopandas.read_parquet("outputs/austria_fields.parquet").plot()` renders polygons
- [ ] Notebook runs end-to-end without errors
