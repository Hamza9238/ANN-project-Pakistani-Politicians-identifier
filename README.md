# Pakistani Politicians Image Classification System

A complete deep learning pipeline for classifying images of 16 prominent Pakistani politicians using Convolutional Neural Networks (CNNs). The system automates data collection, preprocessing, training, and evaluation.

## 🎯 Overview

This project implements an end-to-end image classification system that:
1. **Automatically collects** images from Google, Bing, Wikipedia, and DuckDuckGo
2. **Cleans and filters** images using face detection (Haar Cascades/MTCNN) and duplicate removal (perceptual hashing)
3. **Splits the dataset** into train/val/test sets (75/15/10 ratio)
4. **Trains two CNN models**: ResNet-50 and EfficientNet-B2 with transfer learning
5. **Evaluates thoroughly** with confusion matrices, training curves, classification reports, and misclassified sample analysis

## 👤 The 16 Politicians

| # | Name | Class Label |
|---|------|-------------|
| 1 | Imran Khan | `imran_khan` |
| 2 | Nawaz Sharif | `nawaz_sharif` |
| 3 | Shehbaz Sharif | `shehbaz_sharif` |
| 4 | Asif Ali Zardari | `asif_ali_zardari` |
| 5 | Bilawal Bhutto | `bilawal_bhutto` |
| 6 | Maryam Nawaz | `maryam_nawaz` |
| 7 | Fazlur Rehman | `fazlur_rehman` |
| 8 | Pervez Musharraf | `pervez_musharraf` |
| 9 | Chaudhry Nisar | `chaudhry_nisar` |
| 10 | Aitzaz Ahsan | `aitzaz_ahsan` |
| 11 | Sheikh Rasheed | `sheikh_rasheed` |
| 12 | Murad Ali Shah | `murad_ali_shah` |
| 13 | Sanaullah Zehri | `sanaullah_zehri` |
| 14 | Rana Sanaullah | `rana_sanaullah` |
| 15 | Khurshid Shah | `khurshid_shah` |
| 16 | Ahmed Sharif Chaudhry (DG ISPR) | `ahmed_sharif_chaudhry` |

## 🏗️ Project Structure

```
├── main.py                 # Entry point with CLI arguments
├── download_images.py      # Image collection & cleaning pipeline
├── split_dataset.py        # Dataset splitting (75/15/10)
├── dataset_loader.py       # PyTorch DataLoader with augmentation
├── models.py               # ResNet-50 & EfficientNet-B2 definitions
├── trainer.py              # Training loop with early stopping
├── evaluator.py            # Evaluation, plots, and metrics
├── utils.py                # Constants, helpers, face detection
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── raw_dataset/            # Downloaded raw images
├── cleaned_dataset/        # Face-filtered & deduplicated images
├── dataset/                # Final train/val/test splits
│   ├── train/
│   ├── val/
│   └── test/
├── checkpoints/            # Saved model weights
│   ├── best_resnet50.pth
│   └── best_efficientnet_b2.pth
└── results/                # Evaluation outputs
    ├── final_metrics.csv
    ├── confusion_matrix_resnet50.png
    ├── confusion_matrix_efficientnet_b2.png
    ├── training_curves_resnet50.png
    ├── training_curves_efficientnet_b2.png
    ├── misclassified_resnet50.png
    └── misclassified_efficientnet_b2.png
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Everything (Recommended)

```bash
python main.py --mode all
```

This runs the complete pipeline: collect → split → train → evaluate.

### 3. Run Individual Steps

```bash
# Step 1: Download and clean images
python main.py --mode collect

# Step 2: Split into train/val/test
python main.py --mode split

# Step 3: Train both models
python main.py --mode train

# Step 4: Evaluate and generate plots
python main.py --mode evaluate
```

## 🔧 Technical Details

### Data Augmentation (Training Only)
- Random rotation (±30°)
- Random horizontal flip
- Color jitter (brightness, contrast, saturation, hue)
- Random resized crop (zoom 70-100%)
- Random affine transformations

### Models
| Model | Backbone | Pretrained | Parameters |
|-------|----------|-----------|------------|
| ResNet-50 | torchvision | ImageNet V2 | ~25.5M |
| EfficientNet-B2 | timm | ImageNet | ~9.1M |

### Training Configuration
- **Optimizer**: Adam (lr=0.0001, weight_decay=1e-4)
- **Scheduler**: Cosine Annealing
- **Loss**: Cross Entropy
- **Batch Size**: 32
- **Epochs**: 30 (with early stopping, patience=7)
- **Input Size**: 224×224
- **Normalization**: ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### Auto-Retry for Accuracy
If validation accuracy falls below 90%, the system automatically:
1. Retries with halved learning rate (0.00005) for 40 epochs
2. If still below, retries with lr=0.00002 for 50 epochs

### Fallback Mechanism
If image downloading fails for any politician (bot detection, network issues), synthetic placeholder images are automatically generated to ensure the pipeline never crashes.

## 📊 Expected Results

| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| ResNet-50 | ≥90% | Deeper network, good for complex features |
| EfficientNet-B2 | ≥90% | More efficient, fewer parameters |

Detailed per-class metrics are saved to `results/final_metrics.csv`.

## 📈 Output Files

After running the complete pipeline, check the `results/` folder for:

- **`final_metrics.csv`** — Per-class precision, recall, F1 for both models
- **`confusion_matrix_*.png`** — Heatmap confusion matrices
- **`training_curves_*.png`** — Loss and accuracy plots over epochs
- **`misclassified_*.png`** — Top 5 most confidently wrong predictions
- **`classification_report_*.txt`** — Full sklearn classification reports

## ⚙️ Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for training)
- ~10 GB disk space for dataset and models

## 📝 Libraries Used

`torch`, `torchvision`, `timm`, `icrawler`, `opencv-python`, `mtcnn`, `imagehash`, `Pillow`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `requests`, `beautifulsoup4`, `tqdm`
