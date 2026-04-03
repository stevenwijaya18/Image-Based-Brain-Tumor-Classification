# Phase 0 & 1: Brain Tumor MRI Data Pipeline

This repository contains the data pipeline for the BriscMat Brain Tumour MRI Dataset (2026), focusing strictly on **Phase 0 (Exploratory Data Analysis)** and **Phase 1 (Offline Preprocessing and DataLoader Setup)**. 

No training loops, modeling, or advanced augmentations are included here, ensuring a clean and robust foundation for subsequent experimentation phases.

## File Structure

- `utils_dataset.py`: Core utilities for robustly reading `.mat` files and mappings for labels/views.
- `eda.py`: Scripts for Phase 0. Generates visualizations for spatial overlays, pixel intensities, geometric profiles, and class tabulations.
- `preprocess_mat_dataset.py`: Script to convert raw `.mat` files into `.pt` (PyTorch Tensors) offline to remove bottleneck during training. Creates a unified `metadata.json`.
- `dataset.py`: A custom `torch.utils.data.Dataset` that lazily loads the preprocessed `.pt` files.
- `sanity_check.py`: Verification script to ensure data shapes and label distributions remain consistent before moving to Phase 2.

## How to Run

### 1. Exploratory Data Analysis (Phase 0)
Run `eda.py` to analyze random samples and generate plots. It defaults to looking at the kagglehub cache directory but can be configured in the script.
```bash
python eda.py
```
*Outputs will be saved in the `eda_outputs/` directory.*

### 2. Offline Preprocessing (Phase 1)
Convert the raw `.mat` files into efficient `.pt` formats. 
```bash
python preprocess_mat_dataset.py
```
*Note: This will take several minutes as it iterates through all 6000 samples. It will create a `processed/` directory containing the saved tensors and `metadata.json`.*

### 3. Sanity Check
Verify the preprocessing pipeline and DataLoader integration.
```bash
python sanity_check.py
```

### 4. Using the Dataset in Future Phases
In your Phase 2 experiments, integrate the dataset as follows:

```python
from torch.utils.data import DataLoader
from dataset import BrainTumorDataset

# Load datasets
train_dataset = BrainTumorDataset(metadata_path='processed/metadata.json', mode='train')
test_dataset = BrainTumorDataset(metadata_path='processed/metadata.json', mode='test')

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Normal training loop...
```

## Design Decisions
1. **Offline `.pt` Conversion**: Reading `.mat` files dynamically with `scipy.io` during training introduces significant I/O latency. Preprocessing them into PyTorch Tensors (`.pt`) keeps the training loop GPU-bound rather than CPU/Disk-bound.
2. **Global Metadata**: A single `metadata.json` tracks statistics, label info, splits, and mapped paths. This avoids running `os.listdir` on massive folders and simplifies split filtering.
3. **Data Types**: Images are stored as `float32` immediately to prevent precision loss. Masks are stored as `uint8` to save disk space over `float32` (and converted later if specifically needed by a loss function). Labels and Views are kept as `torch.long`.
