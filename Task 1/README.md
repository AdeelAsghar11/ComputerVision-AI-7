# Skin Lesion Classification: CNN Transfer Learning + Classical ML on ISIC 2019

Comparative study of eight pretrained CNN backbones and seven classical classifiers for skin lesion classification on a five-class subset of the ISIC 2019 dataset. Built for a university coursework assignment (Task 01): compare transfer-learning backbones, extract deep features from the best one and compare classical classifiers on those features, tune the best classifier, and compare computational efficiency across the backbones.

## Overview

- **Task 1:** fine-tune 8 ImageNet-pretrained backbones (AlexNet, VGG16, VGG19, ResNet18, ResNet50, ResNet101, DenseNet121, EfficientNet-B0) end to end on the lesion images -> Table 1
- **Task 2:** extract deep features from the best Task 1 backbone and train 7 classical ML classifiers on them -> Table 2
- **Task 3:** grid-search hyperparameters of the best Task 2 classifier, then compare parameter count, model size, FLOPs, and inference time across 7 of the 8 backbones -> Table 3

## Dataset

[ISIC 2019](https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification) (25,331 images, 8 classes). Per the instructor's guidance, reduced to a 5-class subset to fit the available lab time:

| Code | Class | Images | Share |
|---|---|---:|---:|
| NV | Melanocytic Nevus | 12,875 | 53.2% |
| MEL | Melanoma | 4,522 | 18.7% |
| BCC | Basal Cell Carcinoma | 3,323 | 13.7% |
| BKL | Benign Keratosis | 2,624 | 10.8% |
| AK | Actinic Keratosis | 867 | 3.6% |

**Total: 24,211 images.** Stratified 70/15/15 split -> train 16,947 / val 3,632 / test 3,632. Images resized to 224x224 and cached in memory; training images augmented with random horizontal flip and +/-15 degree rotation. Class imbalance handled with balanced class/sample weighting throughout (loss weights ranged from 0.38 for NV to 5.58 for AK).

## Method (short)

- **Backbones:** final layer replaced with a 5-class linear head; 2-phase schedule, warmup (2 epochs, backbone frozen, lr 1e-3) then fine-tune (6 epochs, all layers unfrozen, lr 1e-4); Adam optimizer, class-weighted cross-entropy loss.
- **Deep features:** best backbone's final layer swapped for identity -> penultimate-layer feature vector (2048-d for ResNet50).
- **Classifiers:** Logistic Regression, Decision Tree, Random Forest (200 trees), KNN (k=5), Linear SVM, RBF-SVM, XGBoost (200 estimators); SVM variants trained on a 6,000-image stratified subsample for tractability, all others on the full training set.
- **Tuning:** 3-fold `GridSearchCV` on the winning classifier (RBF-SVM), scored on accuracy, over `C in {0.1, 1, 10, 100}` and `gamma in {scale, 0.01, 0.001}`.
- **Efficiency:** parameter count, on-disk `state_dict` size, FLOPs (`thop`, 2xMACs convention), and mean single-image inference time (30 timed runs after 5 warmup runs).
- **Metrics:** accuracy, macro precision/recall/F1, macro one-vs-rest AUC, all on a held-out test set.

## Results

### Table 1: Comparison of Transfer Learning Models

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
|---|---:|---:|---:|---:|---:|
| AlexNet | 65.09 | 58.16 | 68.58 | 60.19 | 91.11 |
| VGG16 | 71.37 | 63.05 | 70.00 | 65.08 | 92.57 |
| VGG19 | 64.29 | 58.14 | 64.84 | 55.27 | 89.59 |
| ResNet18 | 76.71 | 67.42 | 75.26 | 70.32 | 94.34 |
| **ResNet50** | **81.19** | 73.37 | 76.41 | 74.56 | 95.91 |
| ResNet101 | 80.89 | 73.92 | 79.86 | 76.36 | 96.17 |
| DenseNet121 | 80.95 | 72.28 | 76.75 | 73.60 | 95.02 |
| EfficientNet-B0 | 81.08 | 73.87 | 77.78 | 75.65 | 95.48 |

**Best model: ResNet50** (81.19% accuracy), though ResNet101 actually posts the best F1 and AUC despite slightly lower accuracy, a sign of class imbalance affecting the ranking. The two VGG variants and AlexNet trail the residual/dense architectures by a wide margin.

### Table 2: Comparison of Different Classifiers (deep features from ResNet50)

| Feature Extractor | Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
|---|---|---:|---:|---:|---:|---:|
| Deep Features | Logistic Regression | 81.77 | 76.79 | 74.75 | 75.69 | 94.95 |
| Deep Features | Decision Tree | 68.03 | 59.20 | 57.75 | 58.43 | 73.90 |
| Deep Features | Random Forest | 79.16 | 79.56 | 66.75 | 71.20 | 94.89 |
| Deep Features | KNN | 82.38 | 79.01 | 74.34 | 75.92 | 91.85 |
| Deep Features | Linear SVM | 82.30 | 78.55 | 75.53 | 76.81 | 95.29 |
| **Deep Features** | **RBF-SVM** | **82.98** | 78.07 | 77.71 | 77.83 | 95.40 |
| Deep Features | XGBoost | 82.90 | 77.24 | 75.35 | 76.18 | 95.61 |

**Best classifier: RBF-SVM** (82.98%), narrowly ahead of XGBoost, KNN, and Linear SVM. Decision Tree is the clear outlier, well behind the rest, consistent with single-tree splits being a poor fit for smooth, high-dimensional CNN features. Notably, this classical pipeline (RBF-SVM on frozen ResNet50 features) slightly beats the best fully fine-tuned end-to-end CNN from Table 1.

### Hyperparameter Tuning (RBF-SVM)

The grid search above was started (`Fitting 3 folds for each of 12 candidates, totalling 36 fits`) but the Colab runtime disconnected before it finished, so no tuned result exists yet. Reporting the default-parameter result rather than a guessed one: **accuracy 82.98%, precision 78.07%, recall 77.71%, F1 77.83%, AUC 95.40%** (identical to the RBF-SVM row above, since that classifier's default fit is what Table 2 already reports). Re-running the existing search cell in the same Colab session should take roughly 15-30 minutes and will fill in the tuned-vs-default comparison.

### Table 3: Computational Efficiency Comparison

| Model | Parameters (M) | Model Size (MB) | FLOPs (G) | Inference Time (ms)* | Accuracy (%) |
|---|---:|---:|---:|---:|---:|
| AlexNet | 57.02 | 217.54 | 1.42 | 23.23 | 65.09 |
| VGG16 | 134.28 | 512.25 | 30.93 | 225.78 | 71.37 |
| VGG19 | 139.59 | 532.51 | 39.26 | 258.85 | 64.29 |
| ResNet18 | 11.18 | 42.72 | 3.65 | 36.34 | 76.71 |
| ResNet50 | 23.52 | 90.02 | 8.26 | 84.57 | 81.19 |
| DenseNet121 | 6.96 | 27.12 | 5.79 | 79.20 | 80.95 |
| **EfficientNet-B0** | **4.01** | **15.60** | **0.83** | 27.86 | 81.08 |

\* **Methodology note:** the notebook's own efficiency cell (Section 7) never ran before the same Colab disconnect, so it produced no output. Parameters, model size, and FLOPs above were independently recomputed here using the exact same model definitions and `thop` methodology as the notebook, they are architecture-level facts and do not depend on what machine computes them. Inference time, however, is hardware-dependent: the figures above were measured on a CPU (no GPU was available where this table was filled in), so they are internally consistent for comparing the 7 models to each other but are **not** directly comparable to a Colab GPU number. Re-run the notebook's existing efficiency cell in Colab (well under 5 minutes) for GPU timings before final submission if exact figures matter.

**EfficientNet-B0** is the standout: roughly 5.9x fewer parameters and 5.8x fewer FLOPs than ResNet50 for almost the same accuracy (81.08% vs. 81.19%), making it the best accuracy-per-resource trade-off of the backbones tested. VGG16/VGG19 are the weakest on every axis: largest, slowest, and not the most accurate.

## Limitations

- Uses a 5-class subset of ISIC 2019 (not the full 8-class dataset), per the instructor's guidance to fit lab time; not directly comparable to 8-class literature results.
- SVM classifiers trained on a 6,000-image subsample rather than the full training set, for tractability.
- RBF-SVM hyperparameter tuning (Section 6 of the notebook) and the efficiency benchmark (Section 7) were interrupted by a Colab disconnect; this README reports what can be honestly reported in their place (see notes above) rather than fabricated numbers. Both sections just need a re-run to complete.
- Confusion matrices (Section 8, bonus) were not generated for the same reason; the code is already in the notebook.

## How to Reproduce

1. Open `ISIC2019_SkinLesion_Task01.ipynb` in Google Colab with a GPU runtime.
2. Run all cells top to bottom (dataset downloads automatically via `kagglehub`, no API key needed).
3. To fill in the two incomplete pieces above, just re-run Section 6 (hyperparameter tuning) and Section 7 (efficiency benchmark) after the earlier cells have populated `table1_df`, `table2_df`, and the trained classifiers.

## Repository Contents

- `ISIC2019_SkinLesion_Task01.ipynb`: full notebook (data loading, Task 1, Task 2, tuning, Task 3, confusion matrices)
- `README.md`: this file
- `Task01_Report_Adeel.docx`: full written report (abstract, methodology, results, discussion)

## Author

Adeel, Final Year BS Artificial Intelligence, COMSATS University Islamabad, Wah Campus
