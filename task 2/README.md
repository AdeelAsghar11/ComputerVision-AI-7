# Lab 02: Effect of Image Filtering on Skin-Lesion Classification

Investigates how 5 spatial-domain image filters affect the classification performance of the top 3 pretrained CNN backbones identified in Lab 1, on the same ISIC 2019 5-class skin-lesion subset. Full results below, all real numbers from the executed notebook (`ISIC2019_SkinLesion_Task02_FilterComparison.ipynb`, 18/18 conditions completed).

## Setup recap

- **Models** (best 3 from Lab 1's Table 1, by accuracy): ResNet50 (81.19%), EfficientNet-B0 (81.08%), DenseNet121 (80.95%).
- **Dataset**: ISIC 2019, same 5-class subset as Lab 1 (`MEL, NV, BCC, BKL, AK`, 24,211 images), used in place of HAM10000 for continuity across the two labs (HAM10000 is the predecessor dataset ISIC 2019 was built on top of; the 5 classes used here already exist under the same names in HAM10000 too). Same stratified 70/15/15 split, seed, and image caching as Lab 1: train 16,947 / val 3,632 / test 3,632.
- **Filters**: Average (mean, 5x5), Gaussian (5x5), Median (5x5), Sharpening (unsharp-style kernel), Sobel (grayscale gradient magnitude, replicated to 3 channels), each compared against a "No Filter" baseline, for 3 models x 6 conditions = 18 runs.
- **Training**: identical across all 18 runs (that's what makes the comparison fair): 1 warmup epoch (backbone frozen, lr 1e-3) + 4 fine-tune epochs (unfrozen, lr 1e-4), Adam, class-weighted loss, batch size 64.
- **Metrics**: accuracy, macro precision/recall/F1 (reported as both "F1-score" and "Macro-F1", the brief's table lists them as separate columns but they're the same quantity), balanced accuracy (average per-class recall), and macro one-vs-rest AUC.

## Results: Table 1 (baseline vs. 5 filters, 3 models)

| Model | Filter | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | Macro-F1 (%) | Balanced Accuracy (%) | AUC (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | No Filter | 79.57 | 70.25 | 75.18 | 72.35 | 72.35 | 75.18 | 94.77 |
| ResNet50 | Average | 74.94 | 64.89 | 72.93 | 67.88 | 67.88 | 72.93 | 93.95 |
| ResNet50 | Gaussian | 76.27 | 66.75 | 75.93 | 70.07 | 70.07 | 75.93 | 94.58 |
| ResNet50 | Median | 77.42 | 69.27 | 73.28 | 70.95 | 70.95 | 73.28 | 94.56 |
| ResNet50 | Sharpening | **80.04** | 71.40 | 77.07 | 73.66 | 73.66 | 77.07 | 95.40 |
| ResNet50 | Sobel | 67.51 | 55.93 | 62.31 | 57.71 | 57.71 | 62.31 | 89.13 |
| EfficientNet-B0 | No Filter | 76.38 | 66.90 | 75.80 | 70.09 | 70.09 | 75.80 | 94.55 |
| EfficientNet-B0 | Average | 75.72 | 64.92 | 71.00 | 67.23 | 67.23 | 71.00 | 93.08 |
| EfficientNet-B0 | Gaussian | 75.96 | 65.72 | 73.64 | 68.60 | 68.60 | 73.64 | 93.57 |
| EfficientNet-B0 | Median | 72.96 | 62.74 | 71.33 | 65.48 | 65.48 | 71.33 | 93.00 |
| EfficientNet-B0 | Sharpening | **79.93** | 71.15 | 76.43 | 73.43 | 73.43 | 76.43 | 95.14 |
| EfficientNet-B0 | Sobel | 67.35 | 55.97 | 64.26 | 58.10 | 58.10 | 64.26 | 89.13 |
| DenseNet121 | No Filter | 75.55 | 66.14 | 76.26 | 69.39 | 69.39 | 76.26 | 94.19 |
| DenseNet121 | Average | 74.56 | 64.35 | 71.98 | 67.30 | 67.30 | 71.98 | 92.64 |
| DenseNet121 | Gaussian | **78.28** | 69.03 | 74.02 | 70.22 | 70.22 | 74.02 | 94.26 |
| DenseNet121 | Median | 74.61 | 64.11 | 73.90 | 66.82 | 66.82 | 73.90 | 93.24 |
| DenseNet121 | Sharpening | 77.84 | 68.03 | 75.86 | 69.60 | 69.60 | 75.86 | 94.24 |
| DenseNet121 | Sobel | 63.99 | 54.30 | 64.56 | 55.82 | 55.82 | 64.56 | 88.49 |

Bolded = each model's best-performing condition. Full training/validation curves, confusion matrices, ROC curves, and the original-vs-filtered example grid are saved by the notebook to `accuracy_curves.png`, `loss_curves.png`, `confusion_matrices.png`, `roc_baseline.png`, and `filter_examples.png` respectively (copy these from your Drive results folder into a `figures/` folder alongside this README if you want them rendered inline here).

## Comparative analysis

**Change from each model's own baseline (percentage points):**

| Model | Filter | Δ Accuracy | Δ Macro-F1 | Δ Balanced Accuracy | Δ AUC |
|---|---|---:|---:|---:|---:|
| ResNet50 | Average | -4.63 | -4.47 | -2.25 | -0.82 |
| ResNet50 | Gaussian | -3.30 | -2.28 | 0.75 | -0.19 |
| ResNet50 | Median | -2.15 | -1.40 | -1.90 | -0.21 |
| ResNet50 | Sharpening | +0.47 | +1.31 | +1.89 | +0.63 |
| ResNet50 | Sobel | -12.06 | -14.64 | -12.87 | -5.64 |
| EfficientNet-B0 | Average | -0.66 | -2.86 | -4.80 | -1.47 |
| EfficientNet-B0 | Gaussian | -0.42 | -1.49 | -2.16 | -0.98 |
| EfficientNet-B0 | Median | -3.42 | -4.61 | -4.47 | -1.55 |
| EfficientNet-B0 | Sharpening | +3.55 | +3.34 | +0.63 | +0.59 |
| EfficientNet-B0 | Sobel | -9.03 | -11.99 | -11.54 | -5.42 |
| DenseNet121 | Average | -0.99 | -2.09 | -4.28 | -1.55 |
| DenseNet121 | Gaussian | +2.73 | +0.83 | -2.24 | +0.07 |
| DenseNet121 | Median | -0.94 | -2.57 | -2.36 | -0.95 |
| DenseNet121 | Sharpening | +2.29 | +0.21 | -0.40 | +0.05 |
| DenseNet121 | Sobel | -11.56 | -13.57 | -11.70 | -5.70 |

**Mean absolute change in accuracy vs. baseline, averaged across the 3 models** (largest effect first): Sobel 10.88, Median 2.17, Gaussian 2.15, Sharpening 2.10, Average 2.09.

**Per-class F1 swing across the 6 conditions, averaged across the 3 models** (largest first): AK 22.72, BKL 17.92, BCC 16.31, MEL 15.26, NV 7.92.

## Questions to Answer

### Which three pretrained models performed best in Lab Activity 1?

ResNet50 (81.19% test accuracy), EfficientNet-B0 (81.08%), and DenseNet121 (80.95%), in that order, from Lab 1's Table 1. The next best (ResNet101 at 80.89%) was close enough that the cut is somewhat arbitrary at the third decimal, but the top 3 by accuracy are unambiguous.

### How does filtering affect each of the three models?

All three models follow a similar shape but not an identical one:

- **ResNet50** (baseline 79.57%): hurt by every smoothing filter (Average -4.63, Gaussian -3.30, Median -2.15), helped slightly by Sharpening (+0.47, its only improvement), and devastated by Sobel (-12.06, its largest drop of the three models).
- **EfficientNet-B0** (baseline 76.38%): mildly hurt by Average/Gaussian/Median (-0.42 to -3.42), helped the most of the three models by Sharpening (+3.55), and badly hurt by Sobel (-9.03, its smallest Sobel drop of the three, but still by far its worst condition).
- **DenseNet121** (baseline 75.55%): the only model actually helped by Gaussian (+2.73) rather than hurt, also helped by Sharpening (+2.29), mildly hurt by Average/Median (-0.94 to -0.99), and just as badly hurt by Sobel (-11.56) as ResNet50.

So the qualitative pattern (smoothing hurts a little, sharpening helps, Sobel hurts a lot) holds for all three, but the exact size of each effect, and whether Gaussian specifically helps or hurts, differs by model.

### Which filter produces the greatest change compared with the unfiltered baseline?

**Sobel**, by a wide margin: mean absolute accuracy change of 10.88 percentage points across the 3 models, roughly 5x larger than any other filter (Median 2.17, Gaussian 2.15, Sharpening 2.10, Average 2.09). It's also the largest change on every other metric (Macro-F1, balanced accuracy, AUC) for every model.

### Does the effect of a filter remain consistent across all three models?

Mostly yes, with one exception. Average, Median, Sharpening, and Sobel each move all 3 models in the *same direction* (Average/Median/Sobel: all negative; Sharpening: all positive). **Gaussian is the exception**: it improves DenseNet121 (+2.73) while hurting EfficientNet-B0 (-0.42) and ResNet50 (-3.30). So 4 of the 5 filters behave consistently across architectures; Gaussian's effect is architecture-dependent.

### Does filtering improve or decrease macro-F1 and balanced accuracy?

Mostly decreases, tracking the accuracy pattern closely: Sobel cuts Macro-F1 by 12-15 points and balanced accuracy by 12-13 points for every model, the worst result on both metrics by a large margin. Average, Median, and (for ResNet50/EfficientNet-B0) Gaussian all decrease both metrics too, typically by 1-5 points. **Sharpening is the one filter that increases Macro-F1 for every model** (+0.21 to +3.34) and increases balanced accuracy for 2 of 3 models (ResNet50 +1.89, EfficientNet-B0 +0.63; DenseNet121 essentially flat at -0.40).

### Which lesion classes are most affected by filtering?

By class, averaging the F1 swing across all 6 conditions and all 3 models: **AK is the most affected class** (22.72-point average swing), followed by BKL (17.92) and BCC (16.31); **NV is the least affected** (7.92), with MEL in between (15.26). This lines up with class frequency: AK is the rarest class in the dataset (867 of 24,211 images, 3.6%) and NV the most common (12,875 images, 53.2%), so AK's decision boundary is the least data-supported and the most sensitive to any distribution shift a filter introduces, while NV's is the most robust.

### Why might smoothing remove useful lesion texture or morphological information?

Mean, Gaussian, and median filters all work by averaging or ranking pixel intensities within a local neighborhood, which is precisely a low-pass operation: it suppresses high spatial-frequency content and keeps only slowly-varying intensity. Dermoscopic diagnosis relies heavily on exactly that high-frequency content, irregular pigment networks, fine vascular structure, sharp or blurred border transitions, so smoothing doesn't just remove noise, it also blurs or erases the fine-grained texture and boundary detail a classifier (human or CNN) would otherwise use to tell one lesion type from another. That this notebook's own results show Average and Median as consistently net-negative for all 3 models is a direct illustration of that cost.

### Why might sharpening or edge detection help or hurt classification?

Sharpening boosts local contrast by amplifying high-frequency detail (the opposite of smoothing), which likely accentuates the same texture and border information that smoothing destroys, without discarding anything else about the image, color and overall intensity are still intact. That's consistent with it being the only filter that helped every model here. Sobel edge detection, by contrast, doesn't just emphasize edges, it converts the image into *only* an edge map, discarding color entirely and converting to grayscale first, and collapsing every flat region (however diagnostically relevant its color or texture) to near-zero. For networks pretrained on natural RGB images, that's a far more destructive transformation than sharpening, which matches Sobel being catastrophically worse than every other condition tested (9 to 12 points of accuracy lost, versus a slight gain for sharpening).

### What is the difference between convolution and correlation?

Cross-correlation slides a kernel across an image and computes a weighted sum using the kernel exactly as given. Convolution first flips the kernel 180 degrees (reverses it in both dimensions) before sliding and summing, so convolution equals correlation with a flipped kernel. For a symmetric kernel, the mean/box kernel or a Gaussian kernel used here, flipping changes nothing, so the two operations produce identical results. For an asymmetric, directional kernel, like the Sobel gradient kernels used for edge detection here, the flip matters and the two operations differ. Worth noting: the "convolutional" layers inside every CNN backbone in this notebook are, strictly speaking, implemented as cross-correlation (no kernel flip), a common and mostly harmless naming looseness in deep learning, since the network simply learns whichever kernel orientation is useful during training. OpenCV's `filter2D` (used here for sharpening) performs correlation directly; `cv2.blur`, `cv2.GaussianBlur`, and `cv2.Sobel` use their own optimized implementations that are equivalent to textbook convolution for their respective (symmetric, or well-defined directional) kernels.

### Based on your results, explain the relationship between classical image processing and deep-learning-based feature extraction.

Classical filters are fixed, hand-designed, purely local operations, the same 3x3 or 5x5 rule applied uniformly everywhere in the image, with no notion of what class or task it's being used for. A CNN backbone, by contrast, *learns* a hierarchy of filters from data, early layers that behave something like edge or blob detectors, later layers combining those into increasingly task-specific, nonlinear representations, all tuned end-to-end for the classification objective. This notebook's results are a fairly direct illustration of what happens when you substitute a hand-designed filter for that learned process rather than let the network learn its own: applying Sobel as *preprocessing*, before the network ever sees the raw image, threw away so much information (color, texture, all flat-region content) that it cost every model 9 to 12 points of accuracy, even though a CNN's own early layers would ordinarily learn something edge-like anyway, but as one part of a much richer learned representation, not as a wholesale replacement for the input. Sharpening tells the same story from the other direction: it re-weights frequency content without discarding any of it, so the network still receives the full image and can still learn from it, and it's the only filter that helped every model. The practical takeaway: classical filtering and deep feature extraction aren't interchangeable, and preprocessing that destroys information can actively work against a network that would otherwise have learned to extract exactly that information itself.

## Limitations

- Uses a 5-class ISIC 2019 subset in place of the brief's HAM10000, for continuity with Lab 1 (see Setup recap for why this doesn't change the classes involved).
- A lighter 1 warmup + 4 fine-tune epoch schedule was used for all 18 runs (versus Lab 1's 2+6) to fit the sweep in one lab session; identical across all 18 conditions, so the *comparison* this lab is testing is unaffected, but absolute accuracy runs a few points below Lab 1's numbers for the same 3 models on their own unfiltered baseline.
- Figures (curves, confusion matrices, ROC, filter examples) are saved by the notebook itself (see the Results section above for filenames) but live in the Google Drive results folder rather than this repository; copy them in if you want them rendered here.

## How to Reproduce

1. Open `ISIC2019_SkinLesion_Task02_FilterComparison.ipynb` in Google Colab with a GPU runtime.
2. Run all cells top to bottom. The dataset downloads automatically via `kagglehub` (no API key needed) and results checkpoint to Google Drive as they complete, so a disconnect only costs whatever run was in flight.
3. Re-running the Baseline/Training cells after any interruption is safe: completed (model, filter) pairs are detected and skipped automatically.

## Repository Contents (Lab 02 folder)

- `ISIC2019_SkinLesion_Task02_FilterComparison.ipynb`: full executed notebook (dataset prep, model loading, baseline, filtering, training, evaluation, visualization, comparative analysis)
- `README.md`: this file

## Author

Adeel, Final Year BS Artificial Intelligence, COMSATS University Islamabad, Wah Campus
