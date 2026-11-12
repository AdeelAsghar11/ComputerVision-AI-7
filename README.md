# Computer Vision & AI Lab Assignments

**Course**: Introduction to Computer Vision

This repository contains a collection of computer vision and machine learning assignments, focusing on image processing, feature extraction, and defect detection. The work is divided into a series of tasks covering different datasets and methodologies.

## 📂 Repository Structure

The repository is organized into the following task folders:

### [Task 1: Basic Analysis](Task%201)
- **Dataset**: ISIC 2019 Skin Lesion dataset
- **Focus**: Initial exploratory data analysis, color space conversions, and basic image handling.
- **Key Files**: `ISIC2019_SkinLesion_Task01 (1).ipynb`, `Task01_Report_Adeel.docx`

### [Task 2: Filter Comparison](task%202)
- **Dataset**: ISIC 2019 Skin Lesion dataset
- **Focus**: Comparing various image filters (e.g., smoothing, noise reduction) and analyzing their effects on skin lesion images.
- **Key Files**: `ISIC2019_SkinLesion_Task02_FilterComparison_1 (1).ipynb`

### [Task 3: Edge Detection](task%203)
- **Dataset**: ISIC 2019 Skin Lesion dataset
- **Focus**: Implementing and comparing different edge detection algorithms (such as Canny, Sobel) to highlight important morphological features of the skin lesions.
- **Key Files**: `ISIC2019_SkinLesion_Task03_EdgeDetection (1).ipynb`

### [Task 4: Boundary Detection](lab%20assignment%201%20-%20task%204)
- **Dataset**: ISIC 2019 Skin Lesion dataset
- **Focus**: Advanced boundary detection and segmentation techniques to cleanly isolate the skin lesions from the background.
- **Key Files**: `ISIC2019_SkinLesion_BoundaryDetection (1).ipynb`

### [Task 5: Industrial Steel Defect Detection (HOG & SVM)](task%205)
- **Dataset**: NEU Steel Surface Defect Database (NEU-DET)
- **Focus**: An end-to-end machine learning pipeline extracting Histogram of Oriented Gradients (HOG) features and using Support Vector Machines (SVM) to detect and classify industrial manufacturing defects (e.g., crazing, inclusions, scratches).
- **Key Features**: Includes a fully interactive real-time Streamlit dashboard (`app.py`) with live webcam support and defect heatmaps.
- **Key Files**: `app.py`, `NEU_SteelDefect_Lab05_HOG_Inspection.ipynb`

### [Task 6: Landmark Recognition & Verification (SIFT & RANSAC)](lab%20assignment%202%20-%20task%206)
- **Dataset**: Wikimedia Commons landmark dataset (7 reference landmarks, held-out known and unknown places)
- **Focus**: SIFT feature detection and description, brute-force k-NN matching with Lowe's ratio test, candidate ranking, RANSAC homography geometric verification, bounding polygon projection, and decision rule calibration for open-set landmark recognition and rejection.
- **Key Features**: Includes an interactive Streamlit web application (`app/app.py`), CLI recognition tool (`app/recognize_cli.py`), benchmark tools, and parity test suite.
- **Key Files**: `app/app.py`, `SIFT_Landmark_Recognition_Lab06_executed.ipynb`, `README.md`

### [Task 8: Lab Mid Exam - HOG vs SIFT Comparison](lab%20mid%20-%20task%208)
- **Dataset**: BloodMNIST dataset (eosinophil, monocyte, neutrophil, platelet)
- **Focus**: Comparative study of HOG + SVM versus dense SIFT + SVM on blood-cell classification, evaluating feature representation, accuracy, and robustness across perturbations (rotation, noise, blur, illumination changes).
- **Key Features**: Hand-calculated vs. code-verified HOG feature lengths, dense 6×6 SIFT grid extraction, linear SVM training, and comprehensive perturbation analysis.
- **Key Files**: `FA23_BAI_047_Adeel_LabMid.ipynb`, `README.md`

### [Lab 9: KilnPulse - Brick Kiln Activity Monitoring (Abstract & Intro)](lab%209%20-%20abstract%20and%20intro)
- **Focus**: Multi-sensor satellite remote sensing (Sentinel-1 SAR radar, Sentinel-2 optical, Landsat 8/9 thermal) combined with Random Forest and Hidden Markov Models (HMM) to monitor individual brick kiln operational status across 12-day windows in Pakistan (Kasur and Sheikhupura districts).
- **Key Features**: Overcomes optical smog/fog limitations using Sentinel-1 C-band radar backscatter; leverages Google Earth Engine for environmental compliance monitoring during winter smog shutdown orders.
- **Key Files**: `kilnpulse_abstract_intro.pdf`, `README.md`

---
*Author: Adeel Asghar*
