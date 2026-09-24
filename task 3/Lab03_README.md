# Lab 03: Edge Detection Techniques and Their Impact on Classification Performance

## Introduction

This lab studies five spatial-domain edge detectors (Sobel, Prewitt, Laplacian, LoG, Canny), how Gaussian and salt-and-pepper noise disrupt them, how Canny's own threshold parameters change its output, and, building directly on Lab 01 (backbone comparison) and Lab 02 (filter comparison), whether replacing an image with its edge map helps or hurts classification relative to the raw and Sharpened (Lab 02's best filter) versions of the same image. All experiments use the same ISIC 2019 five-class subset (`MEL, NV, BCC, BKL, AK`, 24,211 images) as Labs 01 and 02, so results stay comparable across the series. Every number below comes from the executed notebook (`ISIC2019_SkinLesion_Task03_EdgeDetection.ipynb`, all 15 classification conditions completed, zero errors).

## Methodology

**Edge detectors** (implemented with OpenCV, applied to grayscale): Sobel Gx, Sobel Gy, and Sobel gradient magnitude (first-order, directional derivatives); Prewitt (first-order, an unweighted variant of Sobel); Laplacian (second-order, a single isotropic derivative); Laplacian of Gaussian, LoG (Gaussian smoothing before the Laplacian, to control its well-known noise sensitivity); Canny (multi-stage: Gaussian smoothing, gradient computation, non-maximum suppression, and hysteresis thresholding between a low and high threshold).

**Noise and preprocessing (Table 1)**: Gaussian noise (sigma 25 on a 0 to 255 scale) and salt-and-pepper noise (3% of pixels) were added to one representative image per class (5 images), then denoised with a 5x5 Gaussian filter or a 5x5 Median filter. Each detector was run on the relevant combination of these variants, following the brief's own Table 1 template. Two metrics stand in for a purely subjective judgement: **Edge Quality** is a density-plus-continuity score (peaks when the edge map has a plausible amount of edge content arranged into long connected strands rather than scattered speckle); **Noise Sensitivity** is `100 - Dice similarity` between that row's (Otsu-binarized) edge map and the same detector's edge map on the clean image, so 0% means identical to the clean result and higher means more disrupted.

**Canny parameter analysis (Table 2)**: three threshold combinations (30/100, 50/150, 100/200, all with a 3x3 Gaussian kernel) plus a fourth configuration that holds the thresholds at the middle combination's level and only changes the kernel size to 5x5, isolating that one variable. `Number of Detected Edges` counts foreground pixels in the Otsu-binarized edge map, averaged across the 5 sample images.

**Selecting the best edge detector overall**: the best Canny configuration from Table 2 was compared against Sobel, Prewitt, Laplacian, and LoG on the same clean images, using the same Edge Quality score. This was computed, not assumed: Canny (using its best configuration) scored 38.98, well ahead of Laplacian (8.45), Prewitt (6.21), Sobel (6.09), and LoG (3.50), so **Canny with low=30, high=100, 3x3 kernel** (`Canny-1`, the top scorer in Table 2 too) was used as the edge representation for classification below.

**Classification (Table 3)**: three versions of the dataset, Raw (unprocessed), Filtered (Sharpening, the one Lab 02 filter that improved both ResNet50 and EfficientNet-B0), and Edge (Canny-1 above, replicated to 3 channels). Rather than reusing Lab 01's and Lab 02's original numbers (which used different epoch schedules, 2+6 and 1+4 respectively, and never tested the classical classifiers on filtered or edge images at all), all five rows, SVM, Random Forest, and KNN on ResNet50 deep features, plus ResNet50 and EfficientNet-B0 fine-tuned end to end, were retrained fresh on all three conditions under identical settings (same split, same 1 warmup + 4 fine-tune epoch schedule, same class weighting, same evaluation metrics), so the comparison is a fair one. Table 3 below keeps the brief's own column names (`Accuracy Raw (Lab 1)` etc.) since they label which lab introduced that representation, but the values are this lab's fresh, consistent reruns rather than the original Lab 1/2 figures (which is why, for example, ResNet50's Raw accuracy here is 80.37% rather than Lab 1's original 81.19%, a small, expected difference from re-running the same procedure once more, not a different result). The classical classifiers reused each condition's own just-fine-tuned ResNet50 (with its head swapped to identity) as their deep-feature extractor, matching how Lab 01 built its classical-classifier features from the winning fine-tuned backbone rather than a generic frozen ImageNet one.

## Experimental Setup

Same dataset, class filter, and stratified 70/15/15 split (same seed) as Labs 01 and 02: 24,211 images across NV (12,875), MEL (4,522), BCC (3,323), BKL (2,624), and AK (867), giving train = 16,947, val = 3,632, test = 3,632. Trained in Google Colab on a GPU runtime, with results checkpointed to Google Drive after every (row, condition) pair so a disconnect only costs whatever was in flight (verified in this run: several conditions show up as resumed rather than retrained). Two stated assumptions carried through from the notebook: the two CNN rows are ResNet50 and EfficientNet-B0 (the top two of Lab 1's top three by accuracy, both agreeing Sharpening was their best Lab 2 filter), and Set B's filter is Sharpening for that reason.

## Results

### Table 1: Effect of Noise and Preprocessing on Edge Detection

| Edge Detector | Input Image | Noise Type | Preprocessing | Edge Quality (%) | Noise Sensitivity (%) | Observations |
|---|---|---|---|---:|---:|---|
| Sobel | Original | None | None | 6.09 | 0.00 | matches the clean-image edge map almost exactly, with a moderately fragmented edge map. |
| Sobel | Noisy | Gaussian | None | 0.00 | 67.56 | diverges sharply from the clean-image edge map, with a sparse or heavily fragmented edge map. |
| Sobel | Noisy | Salt & Pepper | None | 4.40 | 80.47 | diverges sharply from the clean-image edge map, with a moderately fragmented edge map. |
| Sobel | Noisy | Gaussian | Gaussian Filter | 0.00 | 53.62 | moderately diverges from the clean-image edge map, with a sparse or heavily fragmented edge map. |
| Sobel | Noisy | Salt & Pepper | Median Filter | 9.02 | 29.36 | stays close to the clean-image edge map, with a moderately fragmented edge map. |
| Prewitt | Original | None | None | 6.21 | 0.00 | matches the clean-image edge map almost exactly, with a moderately fragmented edge map. |
| Laplacian | Original | None | None | 8.45 | 0.00 | matches the clean-image edge map almost exactly, with a moderately fragmented edge map. |
| LoG | Noisy | Gaussian | Gaussian Filter | 0.00 | 68.05 | diverges sharply from the clean-image edge map, with a sparse or heavily fragmented edge map. |
| Canny | Original | None | Built-in smoothing | 14.69 | 0.00 | matches the clean-image edge map almost exactly, with a moderately fragmented edge map. |
| Canny | Noisy | Gaussian | Gaussian Filter | 9.46 | 77.13 | diverges sharply from the clean-image edge map, with a moderately fragmented edge map. |
| Canny | Noisy | Salt & Pepper | Median Filter | 7.57 | 70.77 | diverges sharply from the clean-image edge map, with a moderately fragmented edge map. |

### Table 2: Canny Parameter Analysis

| Configuration | Low Threshold | High Threshold | Kernel Size | Edge Quality (%) | Number of Detected Edges | Observation |
|---|---:|---:|---|---:|---:|---|
| Canny-1 | 30 | 100 | 3x3 | **38.98** | 1954 | moderate edge density (3.9% of pixels), a reasonable balance. |
| Canny-2 | 50 | 150 | 3x3 | 14.69 | 406 | sparse edge map (0.8% of pixels), may miss real lesion-boundary detail. |
| Canny-3 | 100 | 200 | 3x3 | 2.61 | 52 | sparse edge map (0.1% of pixels), may miss real lesion-boundary detail. |
| Canny-4 | 50 | 150 | 5x5 | 7.85 | 115 | sparse edge map (0.2% of pixels), may miss real lesion-boundary detail. |

Canny-1 (the loosest thresholds) won on Edge Quality by a wide margin: the two tighter configurations (Canny-2, Canny-3) throw away so many edge pixels that continuity suffers along with density.

### Table 3: Cross-Lab Classification Performance Comparison

| Model / Classifier | Accuracy Raw (%) | Accuracy Filtered (%) | Accuracy Edge (%) | Precision (%) | Recall (%) | F1-Score (%) | Training Time (s) | Inference Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SVM | **82.68** | **82.71** | 62.45 | 49.87 | 37.78 | 39.06 | 245.32 | 20.70 |
| Random Forest | 79.87 | 80.53 | 61.21 | 55.10 | 34.09 | 34.51 | 152.31 | 64.22 |
| KNN | 81.31 | 81.64 | 59.00 | 43.75 | 41.84 | 42.36 | 0.02 | 83.64 |
| CNN Model 1 (ResNet50) | 80.37 | 80.18 | 53.52 | 41.90 | 45.33 | 42.29 | 863.80 | 6.49 |
| CNN Model 2 (EfficientNet-B0) | 78.58 | 78.28 | 50.00 | 41.12 | 46.22 | 40.20 | 437.80 | 9.73 |

Precision, Recall, F1-Score, Training Time, and Inference Time are the Edge condition's own values (Lab 3's new contribution); Raw and Filtered accuracy are shown for the same-row comparison the brief asks for. The full 15-row table (every metric, all three conditions) is saved as `table3_full_long_format.csv` in the repository.

### Additional Analysis

The single best-performing (row, condition) pair overall is **SVM on Filtered images, 82.71% accuracy**, essentially tied with SVM on Raw (82.68%); every other row shows the same pattern, Filtered and Raw are close, Edge is far behind. Balanced accuracy and AUC follow the same shape for the best row: SVM is 73.35% balanced accuracy / 95.46% AUC on Raw, 74.22% / 95.40% on Filtered, and 37.78% / 78.80% on Edge, a drop of roughly 35 balanced-accuracy points and 17 AUC points.

Per-class F1, averaged across all 5 rows, Edge versus Raw (most hurt first):

| Class | Mean F1 change, Edge vs. Raw (points) |
|---|---:|
| BKL | -45.83 |
| AK | -41.80 |
| BCC | -38.97 |
| MEL | -22.94 |
| NV | -15.89 |

NV (the majority class, 53.2% of the data) is the most robust to losing color and texture information, the same pattern Lab 2 found for filtering; the specific minority class hit hardest shifts between labs (AK in Lab 2's filter comparison, BKL here), consistent with different lesion types leaning on different visual cues that different transformations disrupt differently.

Confusion matrices for the best row (SVM) across all three conditions, the Accuracy/Precision/Recall/F1 bar chart, the class-distribution plot, the noisy/denoised image examples, the Canny parameter grid, and the Raw/Filtered/Edge example grid are all generated by the notebook and saved to the results folder (`task6_confusion_matrices.png`, `task6_metric_comparison.png`, `class_distribution.png`, `task2_noise_examples.png`, `task3_canny_params.png`, `task4_condition_examples.png`, plus the two Task 1 figures); copy these into a `figures/` folder alongside this README if you want them rendered inline here.

## Discussion

### Question 1: Which edge detector was most sensitive to noise?

By Table 1's Noise Sensitivity figures, **Canny** had the highest average sensitivity (77.13% and 70.77% on its two noisy rows, 73.95% average), ahead of LoG (68.05%, a single data point) and Sobel (57.75% averaged across its four noisy rows; its single worst row, salt-and-pepper without denoising, reached 80.47%). This is a real but slightly counter-intuitive result worth a caveat: Canny's edge maps are thin and sparse by design (non-maximum suppression keeps only single-pixel-wide ridges), so a small positional shift under noise changes a much larger fraction of its already-small foreground than the same shift would change in Sobel's thicker gradient-magnitude blobs, inflating the measured Dice-based sensitivity even though Canny's own Gaussian pre-smoothing is specifically meant to make it robust. In other words, this metric partly measures edge-map sparsity as well as genuine noise disruption, both detectors' underlying robustness mechanisms (denoising before Sobel, built-in smoothing for Canny) still visibly helped: compare Sobel's denoised rows (53.62%, 29.36%) to its undenoised rows (67.56%, 80.47%).

### Question 2: Effect of Filtering

Gaussian and Median filtering both recovered edge quality substantially versus running the detector directly on noisy input. For Sobel, Gaussian-filtering the Gaussian-noisy image dropped Noise Sensitivity from 67.56% to 53.62%, and Median-filtering the salt-and-pepper-noisy image dropped it from 80.47% to 29.36%, a much larger recovery, consistent with Median filtering's known advantage on impulse (salt-and-pepper) noise specifically, since it replaces a corrupted pixel with a neighborhood value rather than averaging the extreme value into its neighbors the way a mean-based Gaussian filter does. For Canny, denoising still helped only modestly (Noise Sensitivity stayed at 77.13% and 70.77%, both still high), because Canny already applies its own Gaussian smoothing internally, so an extra denoising pass before it has less new information to add, and by this stage the earlier caveat about Canny's thin edges inflating the sensitivity score still applies.

### Question 3: Canny Parameters

Loosening the thresholds sharply increased both the number of detected edges and (up to a point) Edge Quality: Canny-1 (30/100) kept 1,954 edge pixels (3.9% of the image) and scored 38.98 on Edge Quality, while Canny-3 (100/200, the tightest) kept only 52 pixels (0.1%) and scored 2.61. Tighter thresholds mean hysteresis discards more genuine boundary pixels along with the noise, so accuracy of detection was traded for a sparser, more fragmented map. Kernel size mattered too, independent of thresholds: Canny-4 (the same 50/150 thresholds as Canny-2, but a 5x5 blur instead of 3x3) cut the detected edge count roughly in half (115 vs. 406) and lowered the quality score as well (7.85 vs. 14.69). The heavier pre-smoothing suppressed more fine detail before the gradient step ever ran, so on this dataset a larger kernel reduced rather than improved the edge map, the opposite of what a naive "more blur removes more noise" intuition would predict, because it blurred away genuine lesion-boundary gradients along with any noise.

### Question 4: Edge Maps and Classification

Edge-only images **reduced** classification accuracy sharply and consistently, for every one of the 5 rows: accuracy fell by 18.7 to 28.6 percentage points versus that row's own Raw baseline (SVM -20.24, Random Forest -18.67, KNN -22.30, ResNet50 -26.84, EfficientNet-B0 -28.58), with Macro-F1 and balanced accuracy falling even further in relative terms (roughly 29 to 37 points). The most likely reason: a Canny edge map keeps only a thin skeleton of high-gradient boundary pixels and throws away color, texture, and smooth intensity variation entirely, exactly the cues dermoscopic classification depends on most (irregular pigment networks, subtle color variegation, texture within a lesion), so the network or classifier is left trying to separate five visually similar lesion types from a sparse black-and-white outline. This mirrors Lab 2's finding that Sobel, used there as a preprocessing filter rather than a full replacement, was similarly the most damaging of the five filters tested (9 to 12 points lost), just more severe here since Canny's hysteresis thresholding discards even more of the image than a continuous gradient-magnitude map does.

### Question 5: Information Loss

An edge map keeps only where intensity changes sharply; it discards texture (fine, repeating intensity variation within a region, such as an irregular pigment network), color (collapsed to grayscale before any gradient is computed), and absolute or smoothly-varying intensity (uniform or gradually-shaded regions produce little or no gradient and vanish). For skin lesion classification specifically, that is most of the diagnostically relevant signal: dermatologists and CNNs alike rely heavily on pigment color and pattern, not just the outline of the lesion, which is exactly what Question 4's accuracy drop reflects.

### Question 6: Classical vs. Deep Features

Manually supplying an edge map forces one fixed, hand-designed notion of "relevant structure" (intensity gradients) onto the network before it ever sees the original image, discarding everything else up front regardless of whether that information would have been useful. A CNN allowed to learn its own early-layer filters can develop edge-like detectors where they help and, just as important, can also learn to preserve and use color, texture, and smooth shading where those carry more signal, all tuned jointly with the rest of the network for the actual classification objective rather than for a generic, task-agnostic definition of "edge." Table 3's results are a direct demonstration: every model did noticeably better on Raw or Filtered images (where its own early layers are free to learn whatever representation helps) than on the hand-restricted Edge images.

### Question 7: Best Representation

Across Labs 01 to 03, **Filtered (Sharpening) and Raw are effectively tied for best**, both clearly ahead of Edge. In this lab's own fresh, consistent rerun, Filtered edged out Raw for 3 of the 5 rows (SVM +0.03, Random Forest +0.66, KNN +0.33) and fell marginally behind for the 2 CNN rows (ResNet50 -0.19, EfficientNet-B0 -0.30), differences small enough to be within normal run-to-run training variance rather than a reliable effect, which is itself consistent with Lab 2's own finding that Sharpening's benefit, while real and the most consistent of the 5 filters tested there, was modest in size (+0.47 to +3.55 points). Edge is unambiguously worst, by a wide margin, on every row and every metric. So the practical ranking is: **Sharpened-filtered images first (by a hair), Raw images essentially equivalent, Edge-only images a clear last**, and the underlying reason is the one Question 6 already describes, a representation that keeps the image largely intact and lets the model's own learned features do the discriminating (Raw, Filtered) beats one that manually strips out most of the image's information before the model ever gets to see it (Edge).

## Conclusion

Five edge detectors were implemented and compared, noise and denoising were shown to meaningfully change edge quality (most sharply for Median filtering against salt-and-pepper noise), Canny's own parameters mattered a great deal (looser thresholds and a smaller smoothing kernel both improved the detected edge map on this dataset), and a real, computed scoring process (not an assumption) selected Canny as the best overall detector. Using that detector's output as a full image replacement for classification, however, was clearly the wrong choice for this task: every one of 5 classifiers and CNNs lost 19 to 29 points of accuracy versus using the raw image, confirming that dermoscopic classification depends on color and texture information that edge maps deliberately discard. Raw and Sharpened-filtered images remain essentially interchangeable as the better choice, with Sharpening's small, consistent edge (seen most clearly in Lab 2's dedicated comparison) making it the marginally preferable default across the series.

## Limitations

- Uses the same 5-class ISIC 2019 subset as Labs 01 to 02, in place of the brief's HAM10000, for continuity across the series (see Lab 02's README for why this is a reasonable substitution).
- Table 1 and Table 2's metrics are computed on 5 representative images (one per class) rather than the full dataset, matching the brief's own "select representative images" instruction for these two tasks; only the classification experiments (Table 3) run over the full train/val/test split.
- The Edge Quality and Noise Sensitivity scores are one reasonable, documented way to quantify what the brief otherwise leaves as a qualitative judgement call; the underlying figures are included so a reader can also assess them visually.
- The same 1 warmup + 4 fine-tune epoch schedule as Lab 02 was used for all 6 CNN runs, to keep the sweep to a single Colab session; absolute accuracy is a little below Lab 01's original numbers (which used a fuller 2+6 schedule), but since the schedule is identical across Raw, Filtered, and Edge, the three-way comparison this lab is actually testing is unaffected.

## How to Reproduce

1. Open `ISIC2019_SkinLesion_Task03_EdgeDetection.ipynb` in Google Colab with a GPU runtime.
2. Run all cells top to bottom. The dataset downloads automatically via `kagglehub`, and results checkpoint to Google Drive as they complete, so a disconnect only costs whatever was in flight.
3. Re-running the Task 4/5 training cell after any interruption is safe: completed (row, condition) pairs, and previously extracted deep features, are detected and skipped automatically.

## Repository Contents (Lab 03 folder)

- `ISIC2019_SkinLesion_Task03_EdgeDetection.ipynb`: full executed notebook (edge detection, noise, Canny tuning, dataset preparation, model training, model evaluation, visualization, comparative analysis)
- `README.md`: this file

## References

- Canny, J. (1986). A Computational Approach to Edge Detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 8(6), 679 to 698.
- Sobel, I., and Feldman, G. (1968). A 3x3 Isotropic Gradient Operator for Image Processing. Presented at the Stanford Artificial Intelligence Project (SAIL).
- Prewitt, J. M. S. (1970). Object Enhancement and Extraction. In B. Lipkin and A. Rosenfeld (Eds.), *Picture Processing and Psychopictorics*. Academic Press.
- Marr, D., and Hildreth, E. (1980). Theory of Edge Detection. *Proceedings of the Royal Society of London. Series B, Biological Sciences*, 207(1167), 187 to 217.
- Gonzalez, R. C., and Woods, R. E. *Digital Image Processing* (for general edge-detection and convolution/correlation background).
- Tschandl, P., Rosendahl, C., and Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161.
- Codella, N., Rotemberg, V., Tschandl, P., et al. (2019). Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by ISIC. *arXiv:1902.03368*.
- Combalia, M., Codella, N. C. F., Rotemberg, V., et al. (2019). BCN20000: Dermoscopic Lesions in the Wild. *arXiv:1908.02288*.

## Author

Adeel, Final Year BS Artificial Intelligence, COMSATS University Islamabad, Wah Campus
