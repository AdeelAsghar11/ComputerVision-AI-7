# Lab Mid Exam: HOG vs SIFT on Blood-Cell Images

**Registration:** FA23-BAI-047 | Computer Vision Lab

---

## Overview

This report covers the blood-cell classification experiment comparing **HOG + SVM** and **SIFT + SVM** on the BloodMNIST dataset. It covers data preparation, feature extraction, model training, robustness testing, and short written answers.

---

## Personal Settings

Settings are derived from the registration number **FA23-BAI-047**:

| Parameter | Value | Rule |
|---|---|---|
| Seed | 47 | Last two digits of registration |
| HOG cell size | 16 × 16 | Last digit is odd → 16 |
| SIFT grid | 6 × 6 (36 keypoints) | Last digit (7) > 4 → 6×6 |

---

## Q1. Data Setup

Four classes from BloodMNIST were used: **eosinophil**, **monocyte**, **neutrophil**, and **platelet**. Images were resized to 128 × 128 and converted to grayscale before feature extraction.

| Split | Images | Per class |
|---|---|---|
| Train | 400 | 100 |
| Validation | 120 | 30 |
| Test | 120 | 30 |

![Training samples — 4 examples per class](assets/q1_bloodmnist_samples.png)

---

## Q2. HOG + SVM

### Feature Extraction

HOG features were extracted using a 16 × 16 cell size, 9 orientations, and 2 × 2 cell blocks with L2-Hys normalisation.

**Hand-calculated feature length** (image 128 × 128, cell 16 × 16):

- Cells per side = 128 / 16 = **8**
- Blocks per side = 8 − 2 + 1 = **7** → 7 × 7 = 49 blocks
- Values per block = 2 × 2 cells × 9 orientations = **36**
- **Feature length = 49 × 36 = 1764** ✓ (matches code output)

![HOG visualisation — one image per class](assets/q2_hog_features.png)

### Training & Results

The best `C` was selected by validation accuracy over `{0.1, 1, 10}`.

| C | Validation Accuracy |
|---|---|
| 0.1 | 0.7583 |
| 1 | 0.7583 |
| 10 | 0.7583 |

**Chosen C = 0.1** — HOG + SVM (validation): **accuracy = 0.7583, macro-F1 = 0.7567**

| | pred eosinophil | pred monocyte | pred neutrophil | pred platelet |
|---|---|---|---|---|
| **true eosinophil** | 19 | 7 | 4 | 0 |
| **true monocyte** | 6 | 21 | 3 | 0 |
| **true neutrophil** | 6 | 1 | 21 | 2 |
| **true platelet** | 0 | 0 | 0 | 30 |

![HOG + SVM confusion matrix](assets/q2_hog_confusion_matrix.png)

---

## Q3. SIFT + SVM

### Why Free Keypoints Don't Work for an SVM

Standard SIFT returns a varying number of keypoints per image (average 71, min 7, max 179 in the training set). An SVM requires every input to be a fixed-length vector, so variable-count descriptors cannot be used directly.

![SIFT keypoints on highest-keypoint image](assets/q3_sift_keypoints.png)

### Fixed-Grid Solution

A uniform 6 × 6 grid of keypoints (spacing ≈ 21.3 px, size 32) was placed on every image so that exactly 36 descriptors are computed per image.

**Hand-calculated feature length:**

- Grid: 6 × 6 = **36 points**
- Each SIFT descriptor: **128 values**
- **Feature length = 36 × 128 = 4608** ✓ (matches code output)

![SIFT grid keypoints on a training image](assets/q3_sift_grid_keypoints.png)

### Training & Results

| C | Validation Accuracy |
|---|---|
| 0.1 | 0.7333 |
| 1 | 0.7333 |
| 10 | 0.7333 |

**Chosen C = 0.1** — SIFT grid + SVM (validation): **accuracy = 0.7333, macro-F1 = 0.7294**

| | pred eosinophil | pred monocyte | pred neutrophil | pred platelet |
|---|---|---|---|---|
| **true eosinophil** | 17 | 10 | 2 | 1 |
| **true monocyte** | 6 | 19 | 5 | 0 |
| **true neutrophil** | 5 | 1 | 22 | 2 |
| **true platelet** | 0 | 0 | 0 | 30 |

![SIFT + SVM confusion matrix](assets/q3_sift_confusion_matrix.png)

---

## Q4. Robustness (Test Set)

Two perturbations were applied to the test images:
- **Rotation:** 20° clockwise warp
- **Noise:** Gaussian noise with σ = 0.05 added to normalised pixel values

![Test image variants — clean, rotated, noisy](assets/q4_robustness_variants.png)

### Accuracy Table

| Model | Clean | Rotated | Noisy |
|---|---|---|---|
| HOG + SVM | 0.7500 | 0.6083 | 0.2500 |
| SIFT + SVM | 0.7583 | 0.4917 | 0.7500 |

### Accuracy Drop

| Model | Rotation Drop | Noise Drop |
|---|---|---|
| HOG + SVM | 0.1417 | **0.5000** |
| SIFT + SVM | **0.2667** | 0.0083 |

### Was My Prediction Correct?

**Pre-experiment prediction:** HOG would lose more under both rotation and noise.

My prediction was **only half correct**. Under **noise** it was correct: HOG fell from 0.7500 to 0.2500 (a drop of 0.5000, which equals chance for four balanced classes) while SIFT fell only 0.0083, probably because SIFT blurs the image with a Gaussian before measuring gradients, which removes most of the noise. Under **rotation** it was wrong: SIFT lost more (a drop of 0.2667) than HOG (0.1417), suggesting that fixed-grid SIFT descriptors without an orientation assignment are actually more sensitive to a 20° rotation than HOG's gradient histograms.

---

## Q5. Short Answers

**(a) Why does HOG use block normalisation?**

Gradient strength changes with local lighting and contrast, so the same shape could produce very different histogram values in a bright versus dark image. HOG groups cells into overlapping blocks and normalises the values inside each block, so a change in brightness or contrast is mostly cancelled out and the descriptor is more stable across lighting conditions.

**(b) Why is it wrong to choose C using the test set?**

The test set must stand in for new, unseen images. If C is chosen because it scores best on the test set, the model has effectively been tuned to that data, making the reported test accuracy too optimistic and no longer an honest estimate of generalisation. C must be chosen on the validation set, and the test set used only once at the very end.

**(c) Most confused classes and the effect of grayscale**

The two classes confused most were **eosinophil and monocyte** in both models — 13 mutual mistakes for HOG (7 + 6) and 16 for SIFT (10 + 6) — while platelet was classified correctly in all 30 validation images by both models. Eosinophils are primarily distinguished by their bright red-orange granules, and monocytes by their bluish-grey cytoplasm and kidney-shaped nucleus. Converting to grayscale removes this colour information, leaving similar rounded grey cells with a textured nucleus that are harder to separate using gradient-based features alone.

---

## AI Log

| Prompt | What I Changed | How I Checked |
|---|---|---|
| No prompt was given — the AI already had my registration number and knew my workflow from previous labs | No changes were needed; the AI generated the correct settings, feature extraction, and explanations based on prior context | Verified printed shapes, class counts, HOG length (1764) and SIFT length (4608) against hand calculations; inspected sample images, HOG visualisations, and grid keypoints |
