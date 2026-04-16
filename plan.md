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

**Validation** runs automatically after each epoch. See the WandB panel summary below for everything logged.

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
Swept parameters: `lr` (log-uniform 1e-4..1e-2), `loss` (ce/logcoshdice/focal), `backbone` (resnet50/efficientnet-b5), `omega` (boundary class weight, values [0.60, 0.75, 0.85] → `class_weights = [0.05, 0.95-ω, ω]`)

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

## WandB Panel Summary

| Key | What it shows | Frequency |
|-----|---------------|-----------|
| `train/loss` | Training loss | step + epoch |
| `train/lr` | Learning rate | epoch |
| `train/grad_norm` | L2 gradient norm — catches exploding/vanishing gradients | step |
| `val/loss` | Validation loss (monitored for checkpointing) | epoch |
| `val/iou_macro` | Mean IoU across all classes | epoch |
| `val/precision_macro` | Mean precision across all classes | epoch |
| `val/recall_macro` | Mean recall across all classes | epoch |
| `val/pixel_f1/field` | Pixel-level F1 for the crop/field class | epoch |
| `iou/background` | Per-class IoU — background | epoch |
| `iou/field` | Per-class IoU — crop interior | epoch |
| `iou/boundary` | Per-class IoU — boundary (key metric for thin boundaries) | epoch |
| `precision/background` · `precision/field` · `precision/boundary` | Per-class precision | epoch |
| `recall/background` · `recall/field` · `recall/boundary` | Per-class recall | epoch |
| `val/object_precision` | Object-level precision (polygon IoU ≥ 0.5) | epoch |
| `val/object_recall` | Object-level recall | epoch |
| `val/object_f1` | Object-level F1 | epoch |
| `val/corner_consensus` | Corner-crop consistency (model stability at patch edges) | epoch |
| `val/predictions` | 8 sample patches: RGB \| GT \| Pred \| Overlay (batches 0–1) | epoch |
| `val/easy_samples` | 4 lowest-loss val patches | epoch |
| `val/hard_samples` | 4 highest-loss val patches | epoch |
| `val/confidence_distribution` | Histogram of softmax max-scores (model certainty) | epoch |

Run config (logged once at startup via `wandb.config`): `backbone`, `loss`, `lr`, `class_weights`, `num_classes`, `in_channels`.

---

## Dev Run

Use `configs/dwei/dev.yaml` (128 samples, 3 epochs, `limit_train_batches: 16`) for fast iteration on logging:

```bash
bash scripts/dev_run.sh
```

Logs to `logs/FTW-Dev/`, WandB tag `dev`. Switch to the full config when logging looks right.

---

## File Map

| Path | Purpose |
|------|---------|
| `configs/dwei/dev.yaml` | Dev config (128 samples, 3 epochs, WandB tag `dev`) |
| `configs/dwei/3_class/full-ftw.yaml` | Full training config (model, data, WandB logger) |
| `configs/dwei/wandb_sweep.yaml` | WandB sweep definition (lr, loss, backbone, omega) |
| `scripts/dev_run.sh` | Dev train → test pipeline |
| `scripts/ftw_model_fit.py` | Sweep agent entry point |
| `ftw_tools/training/trainers.py` | Trainer — all WandB logging logic lives here |
| `notebooks/visualize_results.ipynb` | Result visualization notebook |
| `outputs/test_metrics.csv` | Test metrics (generated at runtime) |
| `outputs/austria_pred.tif` | Full-scene prediction raster (generated at runtime) |
| `outputs/austria_fields.parquet` | Vectorized field polygons (generated at runtime) |

---

## Verification Checklist

- [ ] `bash scripts/dev_run.sh` completes; `last.ckpt` saved under `logs/FTW-Dev/`
- [ ] WandB run (tag: `dev`) appears under project `ftw-baselines`
- [ ] `val/predictions` shows 4-panel figures (RGB \| GT \| Pred \| Overlay) in Media
- [ ] `val/easy_samples` and `val/hard_samples` appear with loss captions
- [ ] `val/confidence_distribution` renders as a histogram
- [ ] `train/grad_norm` appears as a step-level chart
- [ ] `iou/background` and `iou/boundary` are non-zero scalars in Charts
- [ ] `val/pixel_f1/field` is logged
- [ ] WandB run config shows `backbone`, `loss`, `lr`, `class_weights`
- [ ] `ftw model test` writes `outputs/dev_metrics.csv` with non-zero metrics
- [ ] Full run: `ftw model fit --config configs/dwei/3_class/full-ftw.yaml` trains to completion
- [ ] `outputs/austria_pred.tif` is a valid single-band raster
- [ ] `geopandas.read_parquet("outputs/austria_fields.parquet").plot()` renders polygons
- [ ] Notebook runs end-to-end without errors
