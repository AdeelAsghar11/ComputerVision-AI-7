# test_units.py
"""Fast unit tests (no heavy matching).

Tests cover the helper functions individually so they can run in a few
seconds without loading the full database.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Make the app package importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmark_engine import (
    decide,
    draw_keypoints,
    extract_features,
    good_matches,
    limit_side,
    polygon_check,
    ransac_verify,
)


# ---------------------------------------------------------------------------
# limit_side
# ---------------------------------------------------------------------------

class TestLimitSide:
    def test_shrinks_to_800_on_long_side(self):
        img = np.zeros((1200, 900, 3), np.uint8)
        out = limit_side(img, 800)
        assert max(out.shape[:2]) == 800

    def test_does_not_enlarge_small_image(self):
        img = np.zeros((300, 200, 3), np.uint8)
        out = limit_side(img, 800)
        assert out.shape == img.shape

    def test_exact_800_not_resized(self):
        img = np.zeros((800, 600, 3), np.uint8)
        out = limit_side(img, 800)
        assert out.shape == img.shape

    def test_long_side_becomes_800_landscape(self):
        img = np.zeros((600, 1000, 3), np.uint8)
        out = limit_side(img, 800)
        assert out.shape[1] == 800

    def test_preserves_aspect_ratio(self):
        img = np.zeros((600, 900, 3), np.uint8)
        out = limit_side(img, 800)
        original_ratio = 900 / 600
        out_ratio = out.shape[1] / out.shape[0]
        assert abs(original_ratio - out_ratio) < 0.01


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    @pytest.fixture(scope="class")
    @classmethod
    def sift_and_img(cls, config):
        sift_cfg = config["sift"]
        sift = cv2.SIFT_create(
            nfeatures=sift_cfg["nfeatures"],
            nOctaveLayers=sift_cfg["nOctaveLayers"],
            contrastThreshold=sift_cfg["contrastThreshold"],
            edgeThreshold=sift_cfg["edgeThreshold"],
            sigma=sift_cfg["sigma"],
        )
        img = np.random.RandomState(0).randint(0, 255, (400, 600, 3), dtype=np.uint8)
        return sift, img

    def test_descriptor_shape(self, sift_and_img):
        sift, img = sift_and_img
        feat = extract_features(img, sift)
        assert feat["desc"].shape[1] == 128

    def test_descriptor_dtype_float32(self, sift_and_img):
        sift, img = sift_and_img
        feat = extract_features(img, sift)
        assert feat["desc"].dtype == np.float32

    def test_pt_shape_matches_desc(self, sift_and_img):
        sift, img = sift_and_img
        feat = extract_features(img, sift)
        assert feat["pt"].shape[0] == feat["desc"].shape[0]

    def test_t_sift_is_positive(self, sift_and_img):
        sift, img = sift_and_img
        feat = extract_features(img, sift)
        assert feat["t_sift"] > 0


# ---------------------------------------------------------------------------
# good_matches (self-match)
# ---------------------------------------------------------------------------

class TestGoodMatches:
    def test_self_match_returns_most_keypoints(self, config, engine):
        """A photo matched against itself should pass the ratio test for
        almost all keypoints (the closest match is itself, d1 ~ 0)."""
        lm_dir = engine.data_dir / "landmark_data" / "database" / "Hagia Sophia"
        jpg = sorted(lm_dir.glob("*.jpg"))[0]
        bgr = cv2.imread(str(jpg))
        feat = extract_features(bgr, engine._sift)
        qi, ri = good_matches(feat["desc"], feat["desc"], engine.ratio)
        # At least 90% of keypoints should match themselves
        assert len(qi) >= 0.9 * len(feat["pt"])


# ---------------------------------------------------------------------------
# ransac_verify with too few matches
# ---------------------------------------------------------------------------

class TestRansacVerify:
    def test_zero_inliers_when_fewer_than_8_matches(self, config):
        ransac_cfg = config["ransac"]
        qpt = np.random.rand(5, 2).astype(np.float32)
        rpt = np.random.rand(5, 2).astype(np.float32)
        qi = np.arange(5, dtype=int)
        ri = np.arange(5, dtype=int)
        H, mask = ransac_verify(qpt, rpt, qi, ri, ransac_cfg)
        assert int(mask.sum()) == 0

    def test_returns_homography_for_easy_case(self, config):
        """A perfect planar correspondence should give H and all inliers."""
        ransac_cfg = config["ransac"]
        np.random.seed(42)
        rpt = np.random.rand(20, 2).astype(np.float32) * 400
        # A known homography
        H_true = np.array([[1.1, 0.1, 30], [0.05, 1.0, 20], [0, 0, 1]], np.float32)
        qpt_h = cv2.perspectiveTransform(rpt.reshape(-1, 1, 2), H_true).reshape(-1, 2)
        qi = np.arange(20, dtype=int)
        ri = np.arange(20, dtype=int)
        H, mask = ransac_verify(qpt_h, rpt, qi, ri, ransac_cfg)
        assert H is not None
        assert mask.sum() >= 16  # most inliers


# ---------------------------------------------------------------------------
# polygon_check
# ---------------------------------------------------------------------------

class TestPolygonCheck:
    @pytest.fixture
    def poly_cfg(self, config):
        return config["polygon"]

    def _sensible_quad(self):
        return np.float32([[50, 50], [350, 50], [350, 250], [50, 250]])

    def _bow_tie(self):
        return np.float32([[50, 50], [350, 250], [350, 50], [50, 250]])

    def _tiny_quad(self):
        return np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])

    def _very_stretched(self):
        return np.float32([[0, 0], [4000, 0], [4000, 1], [0, 1]])

    def test_accepts_sensible_quad(self, poly_cfg):
        ok, _ = polygon_check(self._sensible_quad(), (800, 600), poly_cfg)
        assert ok

    def test_rejects_bow_tie(self, poly_cfg):
        ok, reason = polygon_check(self._bow_tie(), (800, 600), poly_cfg)
        assert not ok
        assert "convex" in reason

    def test_rejects_tiny_quad(self, poly_cfg):
        ok, reason = polygon_check(self._tiny_quad(), (800, 600), poly_cfg)
        assert not ok
        assert "tiny" in reason

    def test_rejects_very_stretched(self, poly_cfg):
        ok, reason = polygon_check(self._very_stretched(), (800, 600), poly_cfg)
        assert not ok


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------

class TestDecide:
    def _make_res(self, n_good, n_inliers, runner_inliers=0, landmark="Test"):
        runner = (
            dict(landmark="Other", ref="r2", n_good=5, n_inliers=runner_inliers,
                 inlier_ratio=0.5, poly_ok=False, poly_reason="x", poly=None)
            if runner_inliers is not None else None
        )
        best = dict(landmark=landmark, ref="r1", n_good=n_good, n_inliers=n_inliers,
                    inlier_ratio=(n_inliers / n_good if n_good else 0),
                    poly_ok=True, poly_reason="ok", poly=[[0, 0], [1, 0], [1, 1], [0, 1]])
        verified = [best] + ([runner] if runner else [])
        ranking = [dict(landmark=landmark, ref="r1", n_good=n_good)]
        return dict(n_kp=1000, t_sift=0.1, t_match=1.0, t_ransac=0.1,
                    t_total=1.2, ranking=ranking, verified=verified)

    def test_recognized_when_all_pass(self):
        thr = {"t_good": 12, "t_inliers": 6, "margin": 1.5}
        res = self._make_res(n_good=20, n_inliers=10, runner_inliers=5)
        d = decide(res, thr)
        assert d["decision"] == "RECOGNIZED"
        assert d["predicted"] == "Test"

    def test_rejected_too_few_good(self):
        thr = {"t_good": 12, "t_inliers": 6, "margin": 1.5}
        res = self._make_res(n_good=8, n_inliers=10, runner_inliers=5)
        d = decide(res, thr)
        assert d["decision"] == "REJECTED"
        assert "reliable matches" in d["reason"]

    def test_rejected_too_few_inliers(self):
        thr = {"t_good": 12, "t_inliers": 6, "margin": 1.5}
        res = self._make_res(n_good=20, n_inliers=4, runner_inliers=2)
        d = decide(res, thr)
        assert d["decision"] == "REJECTED"
        assert "RANSAC inliers" in d["reason"]

    def test_rejected_not_ahead_of_runner(self):
        thr = {"t_good": 12, "t_inliers": 6, "margin": 1.5}
        res = self._make_res(n_good=20, n_inliers=8, runner_inliers=7)
        d = decide(res, thr)
        assert d["decision"] == "REJECTED"
        assert "not clearly ahead" in d["reason"]

    def test_empty_verified_gives_rejected(self):
        thr = {"t_good": 12, "t_inliers": 6, "margin": 1.5}
        res = dict(n_kp=0, t_sift=0, t_match=0, t_ransac=0, t_total=0,
                   ranking=[], verified=[])
        d = decide(res, thr)
        assert d["decision"] == "REJECTED"
        assert d["candidate"] is None

    def test_reason_includes_all_failures(self):
        thr = {"t_good": 12, "t_inliers": 6, "margin": 1.5}
        # Fails t_good and t_inliers but passes margin
        res = self._make_res(n_good=5, n_inliers=2, runner_inliers=1)
        d = decide(res, thr)
        assert "reliable matches" in d["reason"]
        assert "RANSAC inliers" in d["reason"]


# ---------------------------------------------------------------------------
# report_text format  (Appendix F line-by-line check)
# ---------------------------------------------------------------------------

class TestReportText:
    def test_report_text_lines(self, engine):
        """Run on the clearest pair and check report text format."""
        q_path = (
            engine.data_dir
            / "landmark_data" / "queries" / "known" / "Hagia Sophia"
            / "hagia_sophia_09.jpg"
        )
        result = engine.recognize(str(q_path))
        lines = result.report_text.splitlines()
        assert lines[0] == "LANDMARK RECOGNITION RESULT"
        assert "Query photo:" in lines[1]
        assert "Keypoints found in the query:" in lines[2]
        assert "Most likely landmark:" in lines[3]
        assert "Reliable matches (ratio test):" in lines[4]
        assert "RANSAC inliers:" in lines[5]
        assert "Runner-up:" in lines[6]
        assert "Localization:" in lines[7]
        assert "Decision:" in lines[8]
        assert "Processing time:" in lines[9]

    def test_recognized_decision_format(self, engine):
        q_path = (
            engine.data_dir
            / "landmark_data" / "queries" / "known" / "Hagia Sophia"
            / "hagia_sophia_09.jpg"
        )
        result = engine.recognize(str(q_path))
        assert result.decision == "RECOGNIZED"
        assert "RECOGNIZED as Hagia Sophia" in result.report_text
