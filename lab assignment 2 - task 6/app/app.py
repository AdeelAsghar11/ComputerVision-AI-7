"""
app.py
======
Tourism landmark information app built with Gradio Blocks.

Runs at http://127.0.0.1:7860 by default. No share link. No analytics.
Photos are processed in memory and never written to disk.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


import cv2
import gradio as gr
import numpy as np

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Make the engine importable
_APP_DIR = Path(__file__).resolve().parent
_LAB_DIR = _APP_DIR.parent
sys.path.insert(0, str(_APP_DIR))
from landmark_engine import LandmarkEngine, RecognitionResult

# ---------------------------------------------------------------------------
# Global engine (loaded once at startup)
# ---------------------------------------------------------------------------

_engine = LandmarkEngine(
    _LAB_DIR,
    config_path=_LAB_DIR / "results" / "landmark_config.json",
    cache_dir=_APP_DIR / ".cache",
)

# Load summary for the About tab
with open(_LAB_DIR / "results" / "results_summary.json", encoding="utf-8") as _fh:
    _SUMMARY = json.load(_fh)

# Load config for default thresholds
_CFG = _engine.config
_DEFAULT_THR = _CFG["thresholds"]

# Collect sample photos for the gallery
_SAMPLE_PHOTOS: list[str] = []
for _lm_dir in sorted((_LAB_DIR / "landmark_data" / "queries" / "known").iterdir()):
    if _lm_dir.is_dir():
        jpgs = sorted(_lm_dir.glob("*.jpg"))
        if jpgs:
            _SAMPLE_PHOTOS.append(str(jpgs[0]))
# Add one unknown-test sample
for _unk_dir in sorted((_LAB_DIR / "landmark_data" / "queries" / "unknown_test").iterdir()):
    if _unk_dir.is_dir():
        jpgs = sorted(_unk_dir.glob("*.jpg"))
        if jpgs:
            _SAMPLE_PHOTOS.append(str(jpgs[0]))
            break


# ---------------------------------------------------------------------------
# Helper: convert BGR numpy array to RGB for Gradio
# ---------------------------------------------------------------------------

def _to_rgb(bgr: np.ndarray | None) -> np.ndarray | None:
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_bgr_from_gradio(img) -> np.ndarray | None:
    """Convert a Gradio image (RGB numpy array or path) to BGR."""
    if img is None:
        return None
    if isinstance(img, np.ndarray):
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.imread(str(img))


# ---------------------------------------------------------------------------
# Recognition callback
# ---------------------------------------------------------------------------

def _run_recognition(
    img,
    t_good: float,
    t_inliers: float,
    margin: float,
    use_custom: bool,
):
    """Called when the user clicks Recognise."""
    bgr = _load_bgr_from_gradio(img)
    if bgr is None:
        no = "Please upload a photo first."
        return (no, None, None, None, None, None, None, no, no, 0)

    thr = {"t_good": int(t_good), "t_inliers": int(t_inliers), "margin": float(margin)}
    result: RecognitionResult = _engine.recognize(bgr, thr=thr)

    # Verdict banner HTML
    if result.decision == "RECOGNIZED":
        banner = (
            f'<div style="background:#1a6b2b;color:#fff;padding:16px 20px;'
            f'border-radius:8px;font-size:1.2em;font-weight:bold;">'
            f'Recognised: {result.predicted}</div>'
        )
    else:
        plain_reason = result.reason
        banner = (
            f'<div style="background:#4a4a4a;color:#ddd;padding:16px 20px;'
            f'border-radius:8px;font-size:1.1em;">'
            f'<b>Not recognised</b><br>'
            f'<span style="font-size:0.9em;">{plain_reason}</span></div>'
        )

    # Item 6: localization message
    cand = result.top3[0] if result.top3 else None
    if cand is None:
        outline_msg = "No candidate was found."
    elif result.decision == "RECOGNIZED" and cand["poly_ok"]:
        outline_msg = "Outline drawn: the outline checks passed."
    else:
        reason = cand["poly_reason"] if not cand["poly_ok"] else "the photo was not recognised"
        outline_msg = f"Outline NOT drawn: {reason}"

    # Top 3 candidates table (Markdown)
    if result.top3:
        tbl = "| Rank | Landmark | Best reference photo | Reliable matches | Inliers | Margin check |\n"
        tbl += "|---:|:---|:---|---:|---:|:---|\n"
        for i, c in enumerate(result.top3):
            runner_in = result.top3[1]["n_inliers"] if len(result.top3) > 1 else 0
            margin_ok = (
                c["n_inliers"] >= _DEFAULT_THR["margin"] * max(runner_in, 1)
                if i == 0 else "-"
            )
            ref_file = c["ref"].split("/")[-1] + ".jpg"
            tbl += f"| {i+1} | {c['landmark']} | {ref_file} | {c['n_good']} | {c['n_inliers']} | {margin_ok} |\n"
    else:
        tbl = "No candidates found."

    timing = (
        f"Done in {result.t_total:.2f} s on this computer "
        f"(SIFT {result.t_sift:.2f} s, matching {result.t_match:.2f} s, "
        f"RANSAC {result.t_ransac:.2f} s)"
    )

    return (
        banner,                           # Item 1: verdict_html
        _to_rgb(result.img_keypoints),    # Item 2: out_keypoints
        _to_rgb(result.img_matches),      # Item 3: out_matches
        result.report_text,               # Items 4-5-7: out_report
        _to_rgb(result.img_outline),      # Item 6: out_outline
        outline_msg,                      # out_outline_msg
        tbl,                              # out_candidates
        timing,                           # out_timing
        result.report_text,               # How it works: how_report
        result.n_kp,                      # How it works: how_kp
    )


# ---------------------------------------------------------------------------
# Build credits summary from results_summary.json and credits.csv
# ---------------------------------------------------------------------------

def _credits_summary() -> str:
    test = _SUMMARY["test"]
    known_pct = round(100 * test["known_correct"] / test["known_queries"], 1)

    # Read credits.csv dynamically
    import csv
    from collections import Counter
    credits_path = _LAB_DIR / "landmark_data" / "credits.csv"
    lic_counts = Counter()
    with open(credits_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            lic = row.get("license", "").strip()
            if lic:
                lic_counts[lic] += 1

    lines = [
        "## About this prototype",
        "",
        f"In the lab test on {test['known_queries']} new photos of the 7 known landmarks "
        f"the system recognised {test['known_correct']} ({known_pct}%), named none wrongly "
        f"and rejected {test['known_rejected']}; it rejected {test['unknown_rejected']} of "
        f"{test['unknown_queries']} photos of places that are not in the database. "
        "It is a teaching prototype: when it is not sure, it says so.",
        "",
        "**Why it may miss a known landmark (likely reasons, not tested):**",
        "- Few reference views per landmark: only 8 photos per place were used.",
        "- Some photos do not show the landmark clearly (interior views, road photos).",
        "- A single flat-surface model (homography) may not fit 3D buildings well.",
        "",
        "**Not recognised does not mean it is not that landmark.** "
        "It means the system did not find enough matching keypoints to be confident.",
        "",
        "**The seven landmarks in the database:**",
        ", ".join(_SUMMARY["landmarks"]),
        "",
        "**Photo source:** Wikimedia Commons (free licences).",
        "",
        "| Licence | Photos |",
        "|:---|---:|",
    ]
    for lic, cnt in lic_counts.most_common():
        lines.append(f"| {lic} | {cnt} |")
    lines.extend([
        "",
        "Author and licence of every photo are in `landmark_data/credits.csv`. "
        "Keep that file with the photos when sharing.",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build the Gradio interface
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Tourism Landmark Finder",
    theme=gr.themes.Soft(primary_hue="teal"),
    css="""
    .verdict-box { border-radius: 8px; padding: 16px; margin-bottom: 8px; }
    .timing-line { color: #888; font-size: 0.85em; margin-top: 8px; }
    .threshold-warning { background: #fff3cd; color: #856404; padding: 8px 12px;
                          border-radius: 6px; font-size: 0.85em; }
    """,
) as demo:

    gr.Markdown(
        "# Tourism Landmark Finder\n"
        "Upload a photo of a famous landmark and the system will try to recognise it. "
        "It uses the SIFT feature pipeline from Lab 6: "
        "find keypoints, describe them, match brute-force, verify with RANSAC, and decide.\n"
        "> **Honest warning:** In the lab test on 28 new photos of 7 known landmarks "
        "the system recognised only 5 (17.9%). It is a teaching prototype, not a product."
    )

    # -----------------------------------------------------------------------
    # Tab 1: Find the landmark
    # -----------------------------------------------------------------------
    with gr.Tab("Find the landmark"):
        with gr.Row():
            # Left column: input
            with gr.Column(scale=1):
                img_input = gr.Image(
                    label="Upload a photo",
                    type="numpy",
                    sources=["upload", "webcam"],
                    elem_id="query_image_input",
                )
                recognise_btn = gr.Button(
                    "Recognise",
                    variant="primary",
                    elem_id="recognise_btn",
                )

                gr.Markdown("**Sample photos (click to load)**")
                sample_gallery = gr.Gallery(
                    value=_SAMPLE_PHOTOS,
                    label="Samples from the database queries",
                    columns=4,
                    height=180,
                    elem_id="sample_gallery",
                )

                with gr.Accordion("Advanced: change thresholds", open=False):
                    gr.Markdown(
                        "The lab's thresholds are read from `results/landmark_config.json`. "
                        "Changing them here does not change the lab's findings."
                    )
                    custom_warning = gr.HTML(visible=False, elem_id="custom_threshold_warning")
                    thr_good = gr.Slider(
                        label="Minimum reliable matches (t_good)",
                        minimum=1, maximum=50,
                        value=_DEFAULT_THR["t_good"],
                        step=1, elem_id="thr_good",
                    )
                    thr_inliers = gr.Slider(
                        label="Minimum RANSAC inliers (t_inliers)",
                        minimum=1, maximum=30,
                        value=_DEFAULT_THR["t_inliers"],
                        step=1, elem_id="thr_inliers",
                    )
                    thr_margin = gr.Slider(
                        label="Margin over runner-up",
                        minimum=1.0, maximum=4.0,
                        value=_DEFAULT_THR["margin"],
                        step=0.1, elem_id="thr_margin",
                    )

            # Right column: results
            with gr.Column(scale=2):
                verdict_html = gr.HTML(
                    label="Result",
                    elem_id="verdict_banner",
                )

                with gr.Tabs():
                    with gr.Tab("1. Query photo"):
                        gr.Markdown(
                            "The photo you uploaded, resized to at most 800 px on the long side."
                        )
                        out_query = gr.Image(
                            label="Query photo",
                            elem_id="out_query",
                        )

                    with gr.Tab("2. SIFT keypoints"):
                        gr.Markdown(
                            "Up to 400 of the strongest keypoints found in the query photo. "
                            "Circle size shows the keypoint scale; the line shows its direction."
                        )
                        out_keypoints = gr.Image(
                            label="Query with SIFT keypoints",
                            elem_id="out_keypoints",
                        )

                    with gr.Tab("3. Matched features"):
                        gr.Markdown(
                            "Reference photo on the left, query on the right. "
                            "Green lines are geometrically consistent matches (RANSAC inliers). "
                            "Red lines failed RANSAC."
                        )
                        out_matches = gr.Image(
                            label="Matched features",
                            elem_id="out_matches",
                        )

                    with gr.Tab("4-5-7. Counts and decision"):
                        gr.Markdown(
                            "The full LANDMARK RECOGNITION RESULT block from the pipeline, "
                            "in the same format as the notebook."
                        )
                        out_report = gr.Textbox(
                            label="Recognition report",
                            lines=12,
                            elem_id="out_report",
                        )

                    with gr.Tab("6. Localized region"):
                        gr.Markdown(
                            "When the system is confident, it draws a red outline "
                            "showing where it thinks the landmark region is in your photo."
                        )
                        out_outline = gr.Image(
                            label="Query with outline",
                            elem_id="out_outline",
                        )
                        out_outline_msg = gr.Textbox(
                            label="Outline status",
                            elem_id="out_outline_msg",
                        )

                out_candidates = gr.Markdown(
                    label="Top 3 candidates",
                    elem_id="out_candidates",
                )
                out_timing = gr.Markdown(
                    elem_id="out_timing",
                    value="",
                )

    # -----------------------------------------------------------------------
    # Tab 2: How it works
    # -----------------------------------------------------------------------
    with gr.Tab("How it works"):
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    "## The six steps\n\n"
                    "1. **Find keypoints.** The photo is shrunk to at most 800 px "
                    "and converted to grayscale. OpenCV SIFT finds up to 1,500 "
                    "distinctive spots, each with a scale and direction.\n\n"
                    "2. **Describe each keypoint with 128 numbers.** "
                    "The gradient directions in a 16-cell grid around the spot "
                    "give a compact description that is stable under brightness changes.\n\n"
                    "3. **Match against every reference photo.** "
                    "For each query descriptor the brute-force matcher finds the "
                    "two closest reference descriptors. Lowe's ratio test "
                    "(distance to closest < 0.75 x distance to second-closest) "
                    "keeps only clear winners. The landmark with the most reliable "
                    "matches in its best reference photo is ranked first.\n\n"
                    "4. **Verify geometrically with RANSAC.** "
                    "RANSAC fits a homography (flat-surface model) between the query "
                    "and the best reference photo of each of the top 3 candidates. "
                    "Matches that agree with the model are inliers. "
                    "The candidate with the most inliers wins.\n\n"
                    "5. **Apply three checks.** "
                    "The winner is accepted only if it has at least "
                    f"{_DEFAULT_THR['t_good']} reliable matches, "
                    f"at least {_DEFAULT_THR['t_inliers']} RANSAC inliers, "
                    f"and at least {_DEFAULT_THR['margin']}x the inliers of the runner-up. "
                    "Otherwise the photo is rejected.\n\n"
                    "6. **Draw the region.** "
                    "When the system accepts the photo, it projects the four corners "
                    "of the reference photo into the query using the homography. "
                    "Four checks confirm the outline makes sense "
                    "(convex, not too tiny or too big, not twisted).\n"
                )

            with gr.Column():
                gr.Markdown("## Counts for the current photo")
                how_kp = gr.Number(label="Keypoints in query", elem_id="how_kp")
                how_report = gr.Textbox(
                    label="Recognition detail",
                    lines=12,
                    elem_id="how_report",
                )

    # -----------------------------------------------------------------------
    # Tab 3: About this prototype
    # -----------------------------------------------------------------------
    with gr.Tab("About this prototype"):
        gr.Markdown(_credits_summary())

    # -----------------------------------------------------------------------
    # State: track whether thresholds are custom
    # -----------------------------------------------------------------------
    is_custom = gr.State(False)

    def _on_threshold_change(good, inliers, margin):
        is_changed = (
            good != _DEFAULT_THR["t_good"]
            or inliers != _DEFAULT_THR["t_inliers"]
            or margin != _DEFAULT_THR["margin"]
        )
        warning_html = (
            '<div class="threshold-warning">'
            "These are not the lab\'s thresholds. Results may differ from the notebook."
            "</div>"
            if is_changed else ""
        )
        return warning_html, is_changed

    thr_good.change(
        _on_threshold_change,
        inputs=[thr_good, thr_inliers, thr_margin],
        outputs=[custom_warning, is_custom],
    )
    thr_inliers.change(
        _on_threshold_change,
        inputs=[thr_good, thr_inliers, thr_margin],
        outputs=[custom_warning, is_custom],
    )
    thr_margin.change(
        _on_threshold_change,
        inputs=[thr_good, thr_inliers, thr_margin],
        outputs=[custom_warning, is_custom],
    )

    # Load sample photo when user clicks gallery
    def _load_sample(evt: gr.SelectData):
        path = _SAMPLE_PHOTOS[evt.index]
        bgr = cv2.imread(path)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    sample_gallery.select(_load_sample, None, img_input)

    # Show query image in panel 1 immediately on upload
    def _show_query(img):
        return img

    img_input.change(_show_query, inputs=[img_input], outputs=[out_query])

    # Main recognition event
    recognise_btn.click(
        fn=_run_recognition,
        inputs=[img_input, thr_good, thr_inliers, thr_margin, is_custom],
        outputs=[
            verdict_html,
            out_keypoints,
            out_matches,
            out_report,
            out_outline,
            out_outline_msg,
            out_candidates,
            out_timing,
            how_report,
            how_kp,
        ],
        show_progress="full",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
