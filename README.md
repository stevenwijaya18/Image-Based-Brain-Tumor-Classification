# Multi-Task U-Net for Brain Tumor MRI Classification & Segmentation

![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Environment](https://img.shields.io/badge/Environment-PyTorch%20%7C%20Jupyter-blue)
![Architecture](https://img.shields.io/badge/Architecture-MultiTask%20UNet-orange)

This repository contains a high-performance deep learning pipeline for the automated analysis of brain tumor MRI scans. Using the BriscMat Brain Tumour MRI Dataset (2026), the system performs simultaneous tumor segmentation and multi-class classification using a specialized Multi-Task U-Net architecture.

## 🚀 Key Features

- Multi-Task Learning: A unified model architecture that solves two problems at once:
    - Segmentation: Pixel-level tumor localization using a U-Net decoder.
    - Classification: Categorizing tumors into four classes (Meningioma, Glioma, Pituitary, or No Tumor).
- Physics-Based Feature Engineering: Integrates Z-score normalization and Gradient/Edge Maps to normalize MRI intensity variations and emphasize tumor boundaries.
- View-Aware Embeddings: Incorporates MRI plane information (Axial, Coronal, Sagittal) into the classification head via learnable embeddings, improving accuracy across different scan orientations.
- Advanced Training Pipeline: Implements Mixed Precision (AMP), Gradient Accumulation, and Cosine Annealing learning rate schedules for stable and efficient training.
- Interpretability: Includes Grad-CAM visualization to identify the features the model uses for classification decisions.

## 📊 Dataset Information

The pipeline uses the BriscMat Brain Tumour MRI Dataset (2026) consisting of 6,000 .mat files.

- Dataset Splits: 5,000 samples for training and 1,000 samples for testing.
- Category Labels:
    - 0: No Tumor
    - 1: Meningioma
    - 2: Glioma
    - 3: Pituitary Tumor
- MRI View Orientation:
    - 1: Axial
    - 2: Coronal
    - 3: Sagittal
- File Content (.mat keys):
    - cjdata.image: Grayscale MRI image (typically 0-255).
    - cjdata.tumorMask: Binary segmentation mask (1 for tumor, 0 for background).
    - cjdata.label: Categorical class label.
    - cjdata.view: MRI view identifier.

You can obtain BriscMat Brain Tumour MRI Dataset from [Kaggle](https://www.kaggle.com/datasets/vivanrv/briscmat-brain-tumour-mri-dataset-2026).

## 🏗️ Architecture: Multi-Task U-Net

The pipeline utilizes a custom Multi-Task U-Net designed for simultaneous spatial and categorical inference.

- Encoder: A hierarchical feature extractor with three levels of double convolutions and max pooling, progressively increasing feature depth from 32 to 128 channels.
- Bottleneck: A dense 256-channel latent space capturing the most abstract representations of the MRI slice.
- Segmentation Head: Decodes bottleneck features back to the spatial dimension using transposed convolutions. Skip connections from the encoder are integrated to recover fine-grained spatial details for precise tumor boundary localization.
- Classification Head: Applies global adaptive average pooling to the bottleneck features, which are then fused with a 16-dimensional learnable View Embedding (representing Axial, Coronal, or Sagittal planes) before passing through a dropout-regularized fully connected network.

## ⚙️ Training Process

The training phase is optimized for stability and performance on medical imaging data:

- Multi-Task Optimization: The model optimizes a joint loss function combining Dice Loss (segmentation) and Weighted Cross-Entropy Loss (classification). The loss weights are balanced to prioritize segmentation accuracy (1.0) alongside classification (0.5).
- Balanced Sampling: Uses a WeightedRandomSampler to address class imbalance in the training set, ensuring the model sees infrequent tumor types more often.
- Efficient Training: 
    - Mixed Precision (FP16): Utilizes torch.amp for faster computation and reduced memory footprint.
    - Gradient Accumulation: Simulates larger batch sizes by accumulating gradients over multiple steps before performing an optimizer update.
    - Gradient Clipping: Prevents exploding gradients by capping the maximum norm.
- Learning Rate Schedule: Employs a Cosine Annealing scheduler to smoothly decay the learning rate, helping the model settle into narrower local minima for better generalization.

## 📈 Performance Results

Based on the latest evaluation on the 1,000-sample test set:

| Metric | Score |
| :--- | :--- |
| Test Dice Score | ~0.54 |
| Test IoU | ~0.43 |
| Classification F1-macro | ~0.88 |

Performance varies slightly depending on hyperparameter tuning and random seeds.

## 💻 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ (CUDA recommended)
- kagglehub, scipy, matplotlib, scikit-learn, tqdm

### Usage
The entire pipeline is consolidated into a single, easy-to-follow Jupyter notebook.
1. Clone the repository.
2. Ensure you have Kaggle credentials configured if necessary (though kagglehub handles public downloads).
3. Open and run brain-tumor-classification.ipynb linearly.

The notebook will handle:
- Automatic Data Acquisition: Downloads the dataset to your local cache.
- Exploratory Data Analysis (EDA): Visualizes spatial overlays and class distributions.
- Feature Engineering Preview: Demonstrates the impact of Edge Maps and Normalization.
- Full Training Loop: Executes the multi-task training with real-time progress bars.
- Comprehensive Evaluation: Generates Confusion Matrices, Best/Worst segmentation plots, and Grad-CAM heatmaps.
