# Tourism Landmark App

A small web app that runs the Lab 6 SIFT recognition pipeline on a photo uploaded by a visitor and shows the seven items the brief asks for.

## What it does

Upload a photo and the system will:

1. Find SIFT keypoints in your photo (up to 1,500 at at most 800 px).
2. Describe each keypoint with a 128-number vector.
3. Match against all 56 reference photos with a brute-force matcher and Lowe's ratio test.
4. Run RANSAC homography on the best 3 candidate landmarks.
5. Apply three checks: at least 12 reliable matches, at least 6 RANSAC inliers, and the winner must have at least 1.5x the inliers of the runner-up.
6. Draw the outline when the photo is recognised and the outline passes 4 geometry checks.
7. Show the result: "Recognised: [name]" or "Not recognised" with the reason.

**Honest summary:** In the lab test on 28 new photos of 7 known landmarks the system recognised 5 (17.9%), named none wrongly and rejected 23. It rejected 11 of 12 photos of unknown places. It is a teaching prototype: when it is not sure, it says so.

The pipeline reproduces the notebook exactly (same code, same settings, same thresholds). Thresholds and SIFT settings are always read from `results/landmark_config.json`; they are never hard-coded in the UI.

## Install and run

From the `lab assignment 2 - task 6` folder:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# or
source .venv/bin/activate       # macOS and Linux

pip install -r app/requirements.txt

python app/app.py               # web app at http://127.0.0.1:7860
```

## CLI

```bash
python app/recognize_cli.py "landmark_data/queries/known/Hagia Sophia/hagia_sophia_09.jpg"
python app/recognize_cli.py "photo.jpg" --save-figure out.png
python app/recognize_cli.py "photo.jpg" --thresholds 12 6 1.5
```

`--thresholds` prints a warning that these are not the lab's thresholds.

## Generate screenshots

```bash
python app/tools/make_screenshots.py
```

Writes five result figures to `assets/`:

- `app_hagia_sophia_recognised.png`: Hagia Sophia held-out photo, recognised with 175 RANSAC inliers.
- `app_petra_viewpoint_recognised.png`: Petra Treasury held-out photo, recognised with about 66 inliers.
- `app_pisa_viewpoint_recognised.png`: Leaning Tower of Pisa held-out photo, recognised with about 15 inliers.
- `app_angkor_rejected.png`: Angkor Wat held-out photo, rejected (best candidate 7 inliers against 6 for runner-up).
- `app_eiffel_unknown_rejected.png`: Eiffel Tower photo, rejected (not in the database).

## Benchmark

```bash
python app/tools/benchmark.py
```

Times 10 of the 40 test photos on your computer. Results are labelled "measured on this computer". The Colab run measured 2.88 s per query on average (SIFT 0.16 s, matching 2.63 s, RANSAC 0.09 s).

## Tests

```bash
pytest app/tests -q                     # all tests (including slow parity tests)
pytest app/tests/test_units.py -v       # fast unit tests only
pytest app/tests/test_parity.py -v -m "not slow"   # skip the 40-photo scan
```

The slow tests (marked `@pytest.mark.slow`) take 2 to 4 minutes each.

## Parity tolerance rule

- If your OpenCV version equals `opencv_version` in `results/landmark_config.json` (4.14.0): decisions and predicted landmarks must be identical for all 40 photos, and reliable matches and inliers must be within 2.
- Otherwise: decisions must be identical for at least 38 of 40 photos, the clearest pair within 5 reliable matches and 8 inliers, and the three Hagia Sophia robustness rows within 10 inliers. Every difference is printed.

## Folder layout

```
app/
├── README.md               this file
├── requirements.txt        pip dependencies
├── landmark_engine.py      the pipeline (no UI)
├── app.py                  Gradio web app
├── recognize_cli.py        command line
├── tools/
│   ├── make_screenshots.py generate result figures
│   └── benchmark.py        time test photos on this computer
├── tests/
│   ├── conftest.py         shared fixtures
│   ├── test_units.py       fast unit tests
│   └── test_parity.py      parity tests against the notebook
└── .cache/                 gitignored: cached reference features
```

## The seven items

| Item | Where in the app |
|---:|:---|
| 1. Query photo | Tab "1. Query photo" |
| 2. SIFT keypoints | Tab "2. SIFT keypoints" |
| 3. Matched features | Tab "3. Matched features" |
| 4. Predicted landmark name | Verdict banner and "4-5-7. Counts and decision" tab |
| 5. Number of reliable matches and RANSAC inliers | "4-5-7. Counts and decision" tab and candidates table |
| 6. Localized region | Tab "6. Localized region" |
| 7. Recognised or rejected message | Verdict banner |

## Privacy

Uploaded photos are processed in memory and never written to disk or sent anywhere. The web app binds to `127.0.0.1` and has no share link and no analytics.
