
# Fields of The World (FTW) — Baselines

Quick-start guide for training, evaluating, and running inference with the FTW field-boundary segmentation toolkit. For full documentation see [ORIGINAL_README.md](ORIGINAL_README.md).

---

## Setup

```bash
# Install uv (skip if already installed)
pip install uv

# Create environment and install all dependencies
uv venv --python 3.12
source .venv/Scripts/activate   # Windows (Git Bash)
# source .venv/bin/activate     # macOS / Linux
uv sync --all-extras --dev

# Authenticate WandB (for experiment tracking)
wandb login
```

Verify:
```bash
ftw --help
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Workflow

### 0. Activate environment

```bash
source .venv/Scripts/activate   # Windows (Git Bash)
```

### 1. Download data

```bash
ftw data download --countries=Austria
```

### 2. Dev run (iterate on logging first)

Before committing to a full training run, use the dev script to
1. debug your environment setup,
2. design and validate your WandB logging on a small subset,


```bash
bash scripts/dev_run.sh
```

This trains with `configs/dwei/dev.yaml`, then immediately runs `ftw model test` and writes results to `outputs/dev_metrics.csv`. WandB runs are tagged `dev` so they're easy to filter out from real runs.

Adjust `configs/dwei/dev.yaml` (`limit_train_batches`, `limit_val_batches`, `max_epochs`) based on needs. 

### 3. Train

Edit `configs/dwei/3_class/full-ftw.yaml` to set your model, data, and training options, then:

```bash
ftw model fit --config configs/dwei/3_class/full-ftw.yaml
```

Checkpoints → `logs/FTW-Release-Full-3-class/`  
WandB project → `ftw-baselines`

Resume from a checkpoint:
```bash
ftw model fit --config configs/dwei/3_class/full-ftw.yaml \
  --ckpt_path logs/FTW-Release-Full-3-class/.../last.ckpt
```

### 3. Hyperparameter sweep (optional)

```bash
wandb sweep configs/dwei/wandb_sweep.yaml
wandb agent <entity>/ftw-baselines/<sweep-id>
```

### 4. Test & evaluate

```bash
mkdir -p outputs
ftw model test \
  --model logs/FTW-Release-Full-3-class/.../last.ckpt \
  --countries austria \
  --model_predicts_3_classes --test_on_3_classes \
  --bootstrap \
  --out outputs/test_metrics.csv
```

Outputs pixel-level IoU / precision / recall and object-level precision / recall / F1 with 95% bootstrap CIs.

### 5. Inference on a full scene

```bash
ftw inference run \
  --model logs/FTW-Release-Full-3-class/.../last.ckpt \
  --input <path-to-sentinel2.tif> \
  --output outputs/austria_pred.tif
```

### 6. Polygonize

```bash
ftw inference polygonize outputs/austria_pred.tif \
  --output outputs/austria_fields.parquet \
  --simplify 15 --min_size 500
```

### 7. Visualize results

```bash
jupyter notebook notebooks/visualize_results.ipynb
```

Shows RGB composites, ground-truth masks, predictions, polygon overlays, and field-size statistics.

---

## Key files

| Path | Purpose |
|------|---------|
| `configs/dwei/dev.yaml` | Dev config (128 samples, 3 epochs) |
| `configs/dwei/3_class/full-ftw.yaml` | Full training config |
| `configs/dwei/wandb_sweep.yaml` | WandB sweep definition |
| `scripts/dev_run.sh` | Dev train → test pipeline |
| `scripts/ftw_model_fit.py` | Sweep agent entry point |
| `notebooks/visualize_results.ipynb` | Result visualization |
| `plan.md` | Full workflow reference with command options |
