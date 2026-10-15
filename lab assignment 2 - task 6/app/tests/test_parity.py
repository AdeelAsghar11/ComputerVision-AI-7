# test_parity.py
"""Parity tests: prove the engine reproduces the notebook's outputs.

All photos are in landmark_data/ so no download is needed.
Tests marked @pytest.mark.slow take two to four minutes.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmark_engine import LandmarkEngine, extract_features, limit_side, load_bgr, transform


# ---------------------------------------------------------------------------
# Version-aware tolerance
# ---------------------------------------------------------------------------

def _tolerance(config: dict) -> dict:
    """Return match/inlier tolerance based on whether CV versions match."""
    local_cv = cv2.__version__
    ref_cv = config.get("opencv_version", "")
    same_version = local_cv == ref_cv
    return {
        "same_version": same_version,
        "decisions_must_match": 40 if same_version else 38,
        "clearest_pair_match_tol": 2 if same_version else 5,
        "clearest_pair_inlier_tol": 2 if same_version else 8,
        "hagia_robustness_inlier_tol": 2 if same_version else 10,
    }


# ---------------------------------------------------------------------------
# Test 1: Database counts
# ---------------------------------------------------------------------------

class TestDatabaseCounts:
    def test_database_photos(self, engine):
        db_root = engine.data_dir / "landmark_data" / "database"
        count = sum(1 for _ in db_root.rglob("*.jpg"))
        assert count == 56, f"Expected 56 database photos, found {count}"

    def test_held_out_photos(self, engine):
        q_root = engine.data_dir / "landmark_data" / "queries" / "known"
        count = sum(1 for _ in q_root.rglob("*.jpg"))
        assert count == 28, f"Expected 28 held-out photos, found {count}"

    def test_unknown_calibration_photos(self, engine):
        q_root = engine.data_dir / "landmark_data" / "queries" / "unknown_calibration"
        count = sum(1 for _ in q_root.rglob("*.jpg"))
        assert count == 12, f"Expected 12 calibration photos, found {count}"

    def test_unknown_test_photos(self, engine):
        q_root = engine.data_dir / "landmark_data" / "queries" / "unknown_test"
        count = sum(1 for _ in q_root.rglob("*.jpg"))
        assert count == 12, f"Expected 12 unknown test photos, found {count}"

    def test_credits_rows(self, engine):
        cred_path = engine.data_dir / "landmark_data" / "credits.csv"
        with open(cred_path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 108, f"Expected 108 credits rows, found {len(rows)}"

    def test_seven_landmarks(self, engine):
        expected = sorted([
            "Angkor Wat", "Big Ben", "Burj Khalifa", "Hagia Sophia",
            "Leaning Tower of Pisa", "Petra Treasury", "Sydney Opera House",
        ])
        db_root = engine.data_dir / "landmark_data" / "database"
        found = sorted(d.name for d in db_root.iterdir() if d.is_dir())
        assert found == expected

    def test_test_results_csv_rows(self, engine):
        csv_path = engine.data_dir / "results" / "test_results.csv"
        with open(csv_path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 40, f"Expected 40 rows in test_results.csv, found {len(rows)}"


# ---------------------------------------------------------------------------
# Test 2: Clearest pair (Hagia Sophia)
# ---------------------------------------------------------------------------

class TestClearestPair:
    def test_clearest_pair_matches_and_inliers(self, engine, config):
        tol = _tolerance(config)
        q_path = (
            engine.data_dir
            / "landmark_data" / "queries" / "known" / "Hagia Sophia"
            / "hagia_sophia_09.jpg"
        )
        result = engine.recognize(str(q_path))
        cand = result.top3[0] if result.top3 else None
        assert cand is not None, "No candidate found"

        expected_matches = 210
        expected_inliers = 175
        match_tol = tol["clearest_pair_match_tol"]
        inlier_tol = tol["clearest_pair_inlier_tol"]

        assert abs(cand["n_good"] - expected_matches) <= match_tol, (
            f"Reliable matches: got {cand['n_good']}, "
            f"expected {expected_matches} +/- {match_tol}"
        )
        assert abs(cand["n_inliers"] - expected_inliers) <= inlier_tol, (
            f"RANSAC inliers: got {cand['n_inliers']}, "
            f"expected {expected_inliers} +/- {inlier_tol}"
        )
        assert result.decision == "RECOGNIZED"
        assert result.predicted == "Hagia Sophia"


# ---------------------------------------------------------------------------
# Test 3: All 40 test photos
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestAllTestPhotos:
    def _load_expected(self, engine) -> list[dict]:
        csv_path = engine.data_dir / "results" / "test_results.csv"
        rows = []
        with open(csv_path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows.append(row)
        return rows

    def _photo_path(self, engine, image_id: str) -> Path:
        """Map image_id (e.g. heldout/Hagia Sophia/hagia_sophia_09) to a path."""
        parts = image_id.split("/")
        if parts[0] == "heldout":
            return (
                engine.data_dir
                / "landmark_data" / "queries" / "known"
                / parts[1] / (parts[2] + ".jpg")
            )
        elif parts[0] == "unk_test":
            return (
                engine.data_dir
                / "landmark_data" / "queries" / "unknown_test"
                / parts[1] / (parts[2] + ".jpg")
            )
        raise ValueError(f"Unknown image_id prefix: {image_id}")

    def test_decisions_match(self, engine, config):
        tol = _tolerance(config)
        expected_rows = self._load_expected(engine)
        differences = []

        for row in expected_rows:
            image_id = row["image_id"]
            path = self._photo_path(engine, image_id)
            result = engine.recognize(str(path))

            exp_decision = row["decision"]
            exp_predicted = row["predicted"] if row["predicted"] else None
            exp_matches = int(row["reliable_matches"])
            exp_inliers = int(row["inliers"])

            got_decision = result.decision
            got_predicted = result.predicted
            got_matches = result.top3[0]["n_good"] if result.top3 else 0
            got_inliers = result.top3[0]["n_inliers"] if result.top3 else 0

            if got_decision != exp_decision or got_predicted != exp_predicted:
                differences.append({
                    "image_id": image_id,
                    "expected_decision": exp_decision,
                    "got_decision": got_decision,
                    "expected_predicted": exp_predicted,
                    "got_predicted": got_predicted,
                    "expected_matches": exp_matches,
                    "got_matches": got_matches,
                    "expected_inliers": exp_inliers,
                    "got_inliers": got_inliers,
                })

        # Print differences for visibility
        if differences:
            print(f"\n{len(differences)} differences found:")
            for d in differences:
                print(
                    f"  {d['image_id']}: "
                    f"expected {d['expected_decision']}/{d['expected_predicted']}, "
                    f"got {d['got_decision']}/{d['got_predicted']}, "
                    f"matches {d['got_matches']} (exp {d['expected_matches']}), "
                    f"inliers {d['got_inliers']} (exp {d['expected_inliers']})"
                )

        n_matching = len(expected_rows) - len(differences)
        required = tol["decisions_must_match"]
        assert n_matching >= required, (
            f"Only {n_matching}/{len(expected_rows)} decisions match "
            f"(need at least {required}). Differences:\n"
            + "\n".join(
                f"  {d['image_id']}: exp {d['expected_decision']}/{d['expected_predicted']}, "
                f"got {d['got_decision']}/{d['got_predicted']}"
                for d in differences
            )
        )


# ---------------------------------------------------------------------------
# Test 4: Hagia Sophia robustness (task6_results.csv)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestHagiaSophiaRobustness:
    """Check rotated, resized and darkened Hagia Sophia against task6_results.csv."""

    def _load_task6(self, engine) -> dict:
        csv_path = engine.data_dir / "results" / "task6_results.csv"
        rows = {}
        with open(csv_path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["landmark"] == "Hagia Sophia":
                    rows[row["condition"]] = row
        return rows

    def test_hagia_sophia_rotated_45(self, engine, config):
        tol = _tolerance(config)
        task6 = self._load_task6(engine)
        row = task6.get("rotated 45 deg")
        assert row is not None

        q_path = (
            engine.data_dir
            / "landmark_data" / "queries" / "known" / "Hagia Sophia"
            / row["query"]
        )
        bgr = load_bgr(q_path, engine.max_side)
        bgr_t = transform(bgr, "rotation", 45.0)
        result = engine.recognize(bgr_t)

        exp_inliers = int(row["ransac_inliers"])
        got_inliers = result.top3[0]["n_inliers"] if result.top3 else 0
        assert abs(got_inliers - exp_inliers) <= tol["hagia_robustness_inlier_tol"], (
            f"Rotated 45: inliers {got_inliers}, expected {exp_inliers} "
            f"+/- {tol['hagia_robustness_inlier_tol']}"
        )
        assert result.decision == "RECOGNIZED"

    def test_hagia_sophia_resized_50(self, engine, config):
        tol = _tolerance(config)
        task6 = self._load_task6(engine)
        row = task6.get("resized 50%")
        assert row is not None

        q_path = (
            engine.data_dir
            / "landmark_data" / "queries" / "known" / "Hagia Sophia"
            / row["query"]
        )
        bgr = load_bgr(q_path, engine.max_side)
        bgr_t = transform(bgr, "scale", 0.5)
        result = engine.recognize(bgr_t)

        exp_inliers = int(row["ransac_inliers"])
        got_inliers = result.top3[0]["n_inliers"] if result.top3 else 0
        assert abs(got_inliers - exp_inliers) <= tol["hagia_robustness_inlier_tol"], (
            f"Resized 50%: inliers {got_inliers}, expected {exp_inliers} "
            f"+/- {tol['hagia_robustness_inlier_tol']}"
        )
        assert result.decision == "RECOGNIZED"

    def test_hagia_sophia_dark(self, engine, config):
        tol = _tolerance(config)
        task6 = self._load_task6(engine)
        row = task6.get("brightness x0.4")
        assert row is not None

        q_path = (
            engine.data_dir
            / "landmark_data" / "queries" / "known" / "Hagia Sophia"
            / row["query"]
        )
        bgr = load_bgr(q_path, engine.max_side)
        bgr_t = transform(bgr, "brightness", 0.4)
        result = engine.recognize(bgr_t)

        exp_inliers = int(row["ransac_inliers"])
        got_inliers = result.top3[0]["n_inliers"] if result.top3 else 0
        assert abs(got_inliers - exp_inliers) <= tol["hagia_robustness_inlier_tol"], (
            f"Brightness x0.4: inliers {got_inliers}, expected {exp_inliers} "
            f"+/- {tol['hagia_robustness_inlier_tol']}"
        )
        assert result.decision == "RECOGNIZED"


# ---------------------------------------------------------------------------
# Test 5: Summary numbers
# ---------------------------------------------------------------------------

class TestSummaryNumbers:
    def test_summary_counts(self, engine):
        summary_path = engine.data_dir / "results" / "results_summary.json"
        with open(summary_path, encoding="utf-8") as fh:
            summary = json.load(fh)
        test = summary["test"]
        assert test["known_queries"] == 28
        assert test["known_correct"] == 5
        assert test["known_wrong"] == 0
        assert test["known_rejected"] == 23
        assert test["unknown_queries"] == 12
        assert test["unknown_rejected"] == 11
