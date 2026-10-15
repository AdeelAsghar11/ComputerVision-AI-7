"""
recognize_cli.py
================
Command-line interface for the tourism landmark app.

Usage:
    python recognize_cli.py "path/to/photo.jpg"
    python recognize_cli.py "path/to/photo.jpg" --save-figure out.png
    python recognize_cli.py "path/to/photo.jpg" --thresholds 12 6 1.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def _find_data_dir() -> Path:
    """Return the lab folder (the parent of app/)."""
    return Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Landmark recognition CLI")
    parser.add_argument("photo", help="Path to the query photo")
    parser.add_argument(
        "--save-figure",
        metavar="OUT.png",
        help="Save a three-panel result figure to this file",
    )
    parser.add_argument(
        "--thresholds",
        nargs=3,
        metavar=("T_GOOD", "T_INLIERS", "MARGIN"),
        type=float,
        help="Override the lab thresholds (t_good, t_inliers, margin). "
             "Prints a warning that these are not the lab's thresholds.",
    )
    args = parser.parse_args()

    data_dir = _find_data_dir()
    config_path = data_dir / "results" / "landmark_config.json"
    cache_dir = data_dir / "app" / ".cache"

    # Import here so startup is fast even when showing --help
    sys.path.insert(0, str(data_dir / "app"))
    from landmark_engine import LandmarkEngine

    engine = LandmarkEngine(data_dir, config_path=config_path, cache_dir=cache_dir)

    thr = None
    if args.thresholds:
        t_good, t_inliers, margin = args.thresholds
        thr = {"t_good": int(t_good), "t_inliers": int(t_inliers), "margin": float(margin)}
        print(
            "WARNING: thresholds overridden by command line. "
            "These are not the lab's thresholds. "
            f"(t_good={int(t_good)}, t_inliers={int(t_inliers)}, margin={float(margin)})"
        )

    result = engine.recognize(args.photo, thr=thr)
    print(result.report_text)

    if args.save_figure:
        import numpy as np
        panels = []
        if result.img_keypoints is not None:
            panels.append(result.img_keypoints)
        if result.img_matches is not None:
            panels.append(result.img_matches)
        if result.img_outline is not None:
            panels.append(result.img_outline)
        if panels:
            # Stack panels horizontally at a common height
            h = max(p.shape[0] for p in panels)
            scaled = []
            for p in panels:
                if p.shape[0] != h:
                    s = h / p.shape[0]
                    p = cv2.resize(p, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                scaled.append(p)
            fig = np.concatenate(scaled, axis=1)
            cv2.imwrite(args.save_figure, fig)
            print(f"Figure saved to {args.save_figure}")


if __name__ == "__main__":
    main()
