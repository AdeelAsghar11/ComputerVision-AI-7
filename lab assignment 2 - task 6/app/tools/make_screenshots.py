"""
make_screenshots.py
===================
Generate result figures for the five sample cases described in Phase 6.
Writes to assets/ in the lab folder.

Usage:
    python tools/make_screenshots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Make the engine importable
_APP_DIR = Path(__file__).resolve().parent.parent
_LAB_DIR = _APP_DIR.parent
sys.path.insert(0, str(_APP_DIR))
from landmark_engine import LandmarkEngine


def _bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _stack_panels(panels: list[np.ndarray], target_h: int = 540) -> np.ndarray:
    """Stack panels horizontally at a common height."""
    scaled = []
    for p in panels:
        if p is None:
            continue
        h = p.shape[0]
        if h != target_h:
            s = target_h / h
            p = cv2.resize(p, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        scaled.append(p)
    return np.concatenate(scaled, axis=1)


def _add_title(img: np.ndarray, title: str) -> np.ndarray:
    """Add a white title bar above the image."""
    bar_h = 36
    bar = np.full((bar_h, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def make_figure(engine: LandmarkEngine, photo_path: Path, label: str, out_path: Path):
    result = engine.recognize(str(photo_path))

    panels = []
    if result.img_keypoints is not None:
        panels.append(result.img_keypoints)
    if result.img_matches is not None:
        panels.append(result.img_matches)
    if result.img_outline is not None:
        panels.append(result.img_outline)

    if not panels:
        print(f"  No panels for {label}")
        return

    fig = _stack_panels(panels)

    # Build a title string
    cand = result.top3[0] if result.top3 else None
    if result.decision == "RECOGNIZED" and cand:
        title_str = (
            f"{label}: RECOGNISED as {result.predicted} "
            f"| matches {cand['n_good']} | inliers {cand['n_inliers']}"
        )
    else:
        reason_short = result.reason[:70] + "..." if len(result.reason) > 70 else result.reason
        title_str = f"{label}: REJECTED - {reason_short}"

    fig = _add_title(fig, title_str)
    cv2.imwrite(str(out_path), fig)
    print(f"  Saved: {out_path.name}")
    # Print the report text
    for line in result.report_text.splitlines():
        print(f"    {line}")


def main():
    engine = LandmarkEngine(
        _LAB_DIR,
        config_path=_LAB_DIR / "results" / "landmark_config.json",
        cache_dir=_APP_DIR / ".cache",
    )
    assets_dir = _LAB_DIR / "assets"
    assets_dir.mkdir(exist_ok=True)

    cases = [
        (
            _LAB_DIR / "landmark_data" / "queries" / "known" / "Hagia Sophia" / "hagia_sophia_09.jpg",
            "Hagia Sophia (known, expected: recognised)",
            assets_dir / "app_hagia_sophia_recognised.png",
        ),
        (
            _LAB_DIR / "landmark_data" / "queries" / "known" / "Petra Treasury" / "petra_treasury_10.jpg",
            "Petra Treasury (known, expected: recognised)",
            assets_dir / "app_petra_viewpoint_recognised.png",
        ),
        (
            _LAB_DIR / "landmark_data" / "queries" / "known" / "Leaning Tower of Pisa" / "leaning_tower_of_pisa_11.jpg",
            "Leaning Tower of Pisa (known, expected: recognised)",
            assets_dir / "app_pisa_viewpoint_recognised.png",
        ),
        (
            _LAB_DIR / "landmark_data" / "queries" / "known" / "Angkor Wat" / "angkor_wat_09.jpg",
            "Angkor Wat (known, expected: rejected)",
            assets_dir / "app_angkor_rejected.png",
        ),
        (
            _LAB_DIR / "landmark_data" / "queries" / "unknown_test" / "Eiffel Tower" / "eiffel_tower_01.jpg",
            "Eiffel Tower (unknown, expected: rejected)",
            assets_dir / "app_eiffel_unknown_rejected.png",
        ),
    ]

    print("Generating screenshots...")
    for photo_path, label, out_path in cases:
        if not photo_path.exists():
            print(f"  SKIPPED (file not found): {photo_path}")
            continue
        print(f"\n{label}")
        make_figure(engine, photo_path, label, out_path)

    print("\nDone. Files written to assets/")


if __name__ == "__main__":
    main()
