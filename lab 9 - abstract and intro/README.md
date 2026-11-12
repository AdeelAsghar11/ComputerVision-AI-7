# KilnPulse: Monitoring Whether Individual Brick Kilns Are Running with Sentinel-1, Sentinel-2 and Landsat

**Authors:** Muhammad Ismail¹, Adeel Asghar¹, Asad Masood¹  
**Affiliation:** ¹Computer Science Department, COMSATS University Islamabad, Wah, Pakistan  

---

## 📄 Abstract

Brick kilns are a major source of air pollution in Pakistan, and Punjab orders them shut during the winter smog season. Maps now show where kilns are, but not which ones are running. We present **KilnPulse**, which estimates for each mapped kiln, in every 12-day window, whether it is running, using only free data in Google Earth Engine. 

Every clue compares a kiln with its own surrounding fields:
- **Sentinel-1 (SAR Radar):** Radar change in the kiln yard, seeing through smog, fog, and cloud cover.
- **Sentinel-2 (Optical/Multispectral):** Vegetation, brick-red, and dark-surface contrasts.
- **Landsat 8/9 (Thermal):** Surface warmth and heat signatures.

A Random Forest combines the multi-sensor clues and a two-state Hidden Markov Model (HMM) smooths each kiln’s temporal sequence. Tested on 300 kilns in Kasur and Sheikhupura districts over two seasons with kiln-windows labelled from very-high-resolution imagery by two annotators (Cohen’s $\kappa$).

**Index Terms:** Brick kilns, activity monitoring, Sentinel-1, Sentinel-2, Landsat, air pollution, hidden Markov model.

---

## 🔍 Key Contributions

1. **KilnPulse Framework:** A per-kiln estimation system determining operational status across 12-day temporal windows using kiln-versus-background contrast in free radar, optical, and thermal imagery.
2. **Multi-Season Labelled Dataset:** A rigorous two-annotator ground truth label set for 300 kilns across two seasons.
3. **Sensor Ablation & Smog Resilience:** Evaluation of individual sensor modalities under heavy winter smog when optical view is obscured, demonstrating the capability of Sentinel-1 radar backscatter.
4. **Temporal Modeling:** District-level time series tracking kiln operations and evaluating adherence to government-mandated winter closure periods.

---

## 📂 Included Files

- `kilnpulse_abstract_intro.pdf`: The abstract and introduction paper draft.
