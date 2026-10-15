"""
benchmark.py
============
Time 10 of the 40 test photos and report SIFT, matching, RANSAC and total.

Usage:
    python tools/benchmark.py

Results are labelled "measured on this computer".
The Colab run measured 2.88 s per query on average
(SIFT 0.16 s, matching 2.63 s, RANSAC 0.09 s).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
_LAB_DIR = _APP_DIR.parent
sys.path.insert(0, str(_APP_DIR))

from landmark_engine import LandmarkEngine


def main():
    engine = LandmarkEngine(
        _LAB_DIR,
        config_path=_LAB_DIR / "results" / "landmark_config.json",
        cache_dir=_APP_DIR / ".cache",
    )

    # Pick the first 10 rows from test_results.csv (the 28 held-out photos)
    csv_path = _LAB_DIR / "results" / "test_results.csv"
    rows = []
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    # Use the first 10
    rows = rows[:10]

    def path_of(image_id: str) -> Path:
        parts = image_id.split("/")
        if parts[0] == "heldout":
            return (
                _LAB_DIR / "landmark_data" / "queries" / "known"
                / parts[1] / (parts[2] + ".jpg")
            )
        return (
            _LAB_DIR / "landmark_data" / "queries" / "unknown_test"
            / parts[1] / (parts[2] + ".jpg")
        )

    sift_times, match_times, ransac_times, total_times = [], [], [], []

    print(f"{'Photo':<50} {'SIFT':>7} {'Match':>7} {'RANSAC':>7} {'Total':>7}")
    print("-" * 80)

    for row in rows:
        p = path_of(row["image_id"])
        if not p.exists():
            print(f"SKIP: {p}")
            continue
        r = engine.recognize(str(p))
        sift_times.append(r.t_sift)
        match_times.append(r.t_match)
        ransac_times.append(r.t_ransac)
        total_times.append(r.t_total)
        print(
            f"{p.name:<50} "
            f"{r.t_sift:>6.2f}s "
            f"{r.t_match:>6.2f}s "
            f"{r.t_ransac:>6.2f}s "
            f"{r.t_total:>6.2f}s"
        )

    import statistics
    print("\nMeasured on this computer:")
    print(f"{'Stage':<20} {'Mean (s)':>10} {'Median (s)':>12}")
    print("-" * 44)
    for name, times in [
        ("SIFT", sift_times),
        ("Matching", match_times),
        ("RANSAC", ransac_times),
        ("Total", total_times),
    ]:
        print(
            f"{name:<20} {statistics.mean(times):>10.3f} {statistics.median(times):>12.3f}"
        )

    print(
        "\nColab reference: Total 2.880 s (SIFT 0.161 s, Matching 2.627 s, RANSAC 0.092 s)"
    )


if __name__ == "__main__":
    main()
