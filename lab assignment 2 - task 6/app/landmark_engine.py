"""
landmark_engine.py
==================
Recognition pipeline for the Lab 6 tourism landmark app.

The numerical logic is taken verbatim from the notebook appendices.
This file contains no UI code. All public functions work on in-memory
arrays and paths; nothing is written to disk except the feature cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np


# ---------------------------------------------------------------------------
# Data class for the recognition result
# ---------------------------------------------------------------------------

@dataclass
class RecognitionResult:
    decision: str                     # "RECOGNIZED" or "REJECTED"
    predicted: str | None             # landmark name when RECOGNIZED
    candidate: str | None             # top candidate landmark name
    reason: str                       # plain-words reason
    count_winner: str | None          # landmark with the most reliable matches
    n_kp: int                         # keypoints in the query
    t_sift: float
    t_match: float
    t_ransac: float
    t_total: float
    top3: list[dict]                  # top 3 verified candidates
    report_text: str                  # the LANDMARK RECOGNITION RESULT block
    img_keypoints: np.ndarray | None  # BGR image of query with keypoints drawn
    img_matches: np.ndarray | None    # BGR side-by-side match drawing
    img_outline: np.ndarray | None    # BGR query with outline (or None)
    ref_bgr: np.ndarray | None        # reference photo (best candidate)
    n_raw: int = 0                    # raw matches against best candidate photo


# ---------------------------------------------------------------------------
# Helper: read BGR, resize, extract features  (Appendix A verbatim)
# ---------------------------------------------------------------------------

def _make_sift(cfg: dict) -> cv2.SIFT:
    return cv2.SIFT_create(
        nfeatures=cfg["nfeatures"],
        nOctaveLayers=cfg["nOctaveLayers"],
        contrastThreshold=cfg["contrastThreshold"],
        edgeThreshold=cfg["edgeThreshold"],
        sigma=cfg["sigma"],
    )


def limit_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    if s > max_side:
        img = cv2.resize(
            img,
            (round(w * max_side / s), round(h * max_side / s)),
            interpolation=cv2.INTER_AREA,
        )
    return img


def load_bgr(path: str | Path, max_side: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read the image {path}")
    return limit_side(img, max_side)


def extract_features(bgr: np.ndarray, sift: cv2.SIFT) -> dict:
    """Steps 1 to 4 of the lab: grayscale, SIFT keypoints, descriptors.
    Keypoints are kept as plain arrays."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    kps, desc = sift.detectAndCompute(gray, None)
    dt = time.perf_counter() - t0
    if desc is None:
        kps, desc = [], np.zeros((0, 128), np.float32)
    return dict(
        pt=np.float32([k.pt for k in kps]).reshape(-1, 2),
        size=np.float32([k.size for k in kps]),
        angle=np.float32([k.angle for k in kps]),
        response=np.float32([k.response for k in kps]),
        octave=np.int32([k.octave for k in kps]),
        desc=desc.astype(np.float32),
        shape=gray.shape,
        t_sift=dt,
    )


def kps_of(feat: dict, idx=None) -> list:
    """Rebuild OpenCV KeyPoint objects (needed only for drawing)."""
    idx = range(len(feat["pt"])) if idx is None else idx
    return [
        cv2.KeyPoint(
            float(feat["pt"][i][0]),
            float(feat["pt"][i][1]),
            float(feat["size"][i]),
            float(feat["angle"][i]),
            float(feat["response"][i]),
            int(feat["octave"][i]),
        )
        for i in idx
    ]


# ---------------------------------------------------------------------------
# Drawing helpers  (Appendix B, D verbatim)
# ---------------------------------------------------------------------------

def draw_keypoints(bgr: np.ndarray, feat: dict, top: int | None = None) -> np.ndarray:
    idx = np.argsort(-feat["response"])[:top] if top else None
    return cv2.drawKeypoints(
        bgr,
        kps_of(feat, idx),
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def _side_by_side(rbgr: np.ndarray, qbgr: np.ndarray):
    h = max(rbgr.shape[0], qbgr.shape[0])
    canvas = np.zeros((h, rbgr.shape[1] + qbgr.shape[1], 3), np.uint8)
    canvas[:rbgr.shape[0], :rbgr.shape[1]] = rbgr
    canvas[:qbgr.shape[0], rbgr.shape[1]:] = qbgr
    return canvas, rbgr.shape[1]


def _ip(p):
    return (int(round(float(p[0]))), int(round(float(p[1]))))


def draw_match_lines(
    rbgr: np.ndarray,
    qbgr: np.ndarray,
    rpt: np.ndarray,
    qpt: np.ndarray,
    ri: np.ndarray,
    qi: np.ndarray,
    inlier_mask: np.ndarray,
    max_lines: int = 160,
) -> np.ndarray:
    """Reference on the left, query on the right.
    Green = RANSAC inlier, red = outlier."""
    canvas, off = _side_by_side(rbgr, qbgr)
    idx = np.arange(len(qi))
    if len(idx) > max_lines:
        idx = np.random.RandomState(0).choice(idx, max_lines, replace=False)
    for j in idx:
        p1 = _ip(rpt[ri[j]])
        p2 = _ip(qpt[qi[j]] + np.array([off, 0]))
        col = (0, 220, 0) if inlier_mask[j] else (0, 0, 255)
        cv2.line(canvas, p1, p2, col, 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, col, -1)
        cv2.circle(canvas, p2, 3, col, -1)
    return canvas


# ---------------------------------------------------------------------------
# Matching  (Appendix C verbatim)
# ---------------------------------------------------------------------------

_BF = cv2.BFMatcher(cv2.NORM_L2)


def knn_two(dq: np.ndarray, dr: np.ndarray):
    """For every query descriptor: its two closest reference descriptors.
    Returns (query idx, reference idx, d1, d2)."""
    if len(dq) == 0 or len(dr) < 2:
        z = np.zeros(0)
        return z.astype(int), z.astype(int), z.astype(np.float32), z.astype(np.float32)
    qi, ri, d1, d2 = [], [], [], []
    for pair in _BF.knnMatch(dq, dr, k=2):
        if len(pair) == 2:
            qi.append(pair[0].queryIdx)
            ri.append(pair[0].trainIdx)
            d1.append(pair[0].distance)
            d2.append(pair[1].distance)
    return (
        np.array(qi, int),
        np.array(ri, int),
        np.array(d1, np.float32),
        np.array(d2, np.float32),
    )


def good_matches(dq: np.ndarray, dr: np.ndarray, ratio: float) -> tuple:
    """Reliable matches = those that pass Lowe's ratio test."""
    qi, ri, d1, d2 = knn_two(dq, dr)
    keep = d1 < ratio * d2
    return qi[keep], ri[keep]


def landmark_ranking(counts_by_ref: dict, lm_of: dict) -> list:
    """Landmark score = reliable matches of its best reference photo.
    Returns [(landmark, best ref id, count)] sorted high to low."""
    best = {}
    for rid, n in counts_by_ref.items():
        lm = lm_of[rid]
        if lm not in best or n > best[lm][1]:
            best[lm] = (rid, n)
    return sorted(
        ((lm, rid, n) for lm, (rid, n) in best.items()),
        key=lambda t: -t[2],
    )


# ---------------------------------------------------------------------------
# RANSAC, outline checks  (Appendix D verbatim)
# ---------------------------------------------------------------------------

def ransac_verify(
    qpt: np.ndarray,
    rpt: np.ndarray,
    qi: np.ndarray,
    ri: np.ndarray,
    ransac_cfg: dict,
):
    """Homography reference -> query with RANSAC.
    Returns (H or None, boolean inlier mask over the given matches)."""
    n = len(qi)
    if n < ransac_cfg["min_matches"]:
        return None, np.zeros(n, bool)
    src = rpt[ri].reshape(-1, 1, 2).astype(np.float32)
    dst = qpt[qi].reshape(-1, 1, 2).astype(np.float32)
    H, mask = cv2.findHomography(
        src,
        dst,
        cv2.RANSAC,
        ransac_cfg["reproj_threshold"],
        maxIters=ransac_cfg["max_iters"],
        confidence=ransac_cfg["confidence"],
    )
    if H is None or mask is None:
        return None, np.zeros(n, bool)
    return H, mask.ravel().astype(bool)


def project_corners(H: np.ndarray, ref_shape: tuple) -> np.ndarray:
    h, w = ref_shape
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(c, H).reshape(-1, 2)


def _interior_angles(poly: np.ndarray) -> list:
    out = []
    for i in range(4):
        a = poly[i - 1] - poly[i]
        b = poly[(i + 1) % 4] - poly[i]
        c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        out.append(float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
    return out


def polygon_check(poly: np.ndarray, q_shape: tuple, poly_cfg: dict) -> tuple:
    """Is the projected outline believable?
    Returns (ok, reason). Four plain-sense checks."""
    if not np.all(np.isfinite(poly)):
        return False, "the corners are not finite numbers"
    contour = poly.astype(np.float32).reshape(-1, 1, 2)
    area = float(cv2.contourArea(contour))
    area_q = float(q_shape[0] * q_shape[1])
    if not cv2.isContourConvex(contour):
        return False, "the outline is twisted (not convex)"
    if area < poly_cfg["min_area_frac"] * area_q:
        return False, "the outline is tiny"
    if area > poly_cfg["max_area_frac"] * area_q:
        return False, "the outline is far larger than the photo"
    sides = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
    if sides.max() / max(sides.min(), 1e-6) > poly_cfg["max_side_ratio"]:
        return False, "the outline is extremely stretched"
    ang = _interior_angles(poly)
    if min(ang) < poly_cfg["min_angle"] or max(ang) > poly_cfg["max_angle"]:
        return False, "the outline has extreme corner angles"
    return True, "ok"


# ---------------------------------------------------------------------------
# Decision rule  (Appendix E verbatim)
# ---------------------------------------------------------------------------

def decide(res: dict, thr: dict) -> dict:
    """Apply the three checks.
    thr = {"t_good", "t_inliers", "margin"}."""
    v = res["verified"]
    count_winner = res["ranking"][0]["landmark"] if res["ranking"] else None
    if not v:
        return dict(
            decision="REJECTED",
            predicted=None,
            candidate=None,
            reason="no candidate",
            count_winner=count_winner,
        )
    best = v[0]
    runner = v[1] if len(v) > 1 else None
    why = []
    if best["n_good"] < thr["t_good"]:
        why.append(f"only {best['n_good']} reliable matches (need {thr['t_good']})")
    if best["n_inliers"] < thr["t_inliers"]:
        why.append(f"only {best['n_inliers']} RANSAC inliers (need {thr['t_inliers']})")
    if runner is not None and best["n_inliers"] < thr["margin"] * max(runner["n_inliers"], 1):
        why.append(
            f"not clearly ahead of {runner['landmark']} "
            f"({best['n_inliers']} against {runner['n_inliers']} inliers, need x{thr['margin']})"
        )
    ok = not why
    return dict(
        decision="RECOGNIZED" if ok else "REJECTED",
        predicted=best["landmark"] if ok else None,
        candidate=best["landmark"],
        reason="all checks passed" if ok else "; ".join(why),
        count_winner=count_winner,
    )


# ---------------------------------------------------------------------------
# Helper transform functions used for parity tests  (Appendix G verbatim)
# ---------------------------------------------------------------------------

def rotate_bound(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    c = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(c, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - c[0]
    M[1, 2] += nh / 2 - c[1]
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def resize_by(img: np.ndarray, s: float) -> np.ndarray:
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, None, fx=s, fy=s, interpolation=interp)


def darken(img: np.ndarray, f: float) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=f, beta=0)


def transform(img: np.ndarray, family: str, value: float) -> np.ndarray:
    return {"rotation": rotate_bound, "scale": resize_by, "brightness": darken}[family](img, value)


# ---------------------------------------------------------------------------
# LandmarkEngine: the main class
# ---------------------------------------------------------------------------

class LandmarkEngine:
    """Loads config and database, caches reference features, and runs the pipeline.

    Parameters
    ----------
    data_dir:
        Path to the lab folder (the one that contains landmark_data/ and results/).
    config_path:
        Path to landmark_config.json. Defaults to <data_dir>/results/landmark_config.json.
    cache_dir:
        Where to store the cached reference features. Defaults to <data_dir>/app/.cache.
    """

    def __init__(
        self,
        data_dir: str | Path,
        config_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.data_dir = Path(data_dir).resolve()
        cfg_path = Path(config_path) if config_path else self.data_dir / "results" / "landmark_config.json"
        with open(cfg_path, encoding="utf-8") as fh:
            self.config = json.load(fh)

        self.sift_cfg = self.config["sift"]
        self.ransac_cfg = self.config["ransac"]
        self.poly_cfg = self.config["polygon"]
        self.thr = self.config["thresholds"]
        self.ratio = float(self.config["ratio"])
        self.verify_top = int(self.config["verify_top_landmarks"])
        self.max_side = int(self.sift_cfg["max_side"])

        self._sift = _make_sift(self.sift_cfg)

        # Scan database to build REF_IDS and LM_OF
        db_root = self.data_dir / "landmark_data" / "database"
        self.ref_ids: list[str] = []
        self.lm_of: dict[str, str] = {}
        self.path_of: dict[str, Path] = {}

        for lm_dir in sorted(db_root.iterdir()):
            if not lm_dir.is_dir():
                continue
            lm = lm_dir.name
            for jpg in sorted(lm_dir.glob("*.jpg")):
                rid = f"ref/{lm}/{jpg.stem}"
                self.ref_ids.append(rid)
                self.lm_of[rid] = lm
                self.path_of[rid] = jpg

        # Build or load feature cache
        cache_root = Path(cache_dir) if cache_dir else self.data_dir / "app" / ".cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        self._feats = self._load_or_build_cache(cache_root)

    # ------------------------------------------------------------------
    # Feature cache
    # ------------------------------------------------------------------

    def _cache_key(self) -> str:
        """A string that changes whenever settings or the file list changes."""
        file_list = [
            (rid, os.path.getsize(self.path_of[rid]))
            for rid in self.ref_ids
        ]
        payload = json.dumps(
            {
                "sift": self.sift_cfg,
                "cv2": cv2.__version__,
                "files": file_list,
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    def _load_or_build_cache(self, cache_root: Path) -> dict:
        key = self._cache_key()
        cache_file = cache_root / f"feats_{key}.joblib"
        if cache_file.exists():
            return joblib.load(cache_file)
        print(f"Building reference feature cache (first run, about 30 s)...")
        feats = {}
        for rid in self.ref_ids:
            bgr = load_bgr(self.path_of[rid], self.max_side)
            feats[rid] = extract_features(bgr, self._sift)
        joblib.dump(feats, cache_file)
        print(f"Cache saved to {cache_file}")
        return feats

    # ------------------------------------------------------------------
    # Process query input
    # ------------------------------------------------------------------

    def _process_query(self, source: str | Path | np.ndarray) -> tuple[np.ndarray, dict]:
        """Load (or resize) a query and extract its features."""
        if isinstance(source, (str, Path)):
            bgr = load_bgr(source, self.max_side)
        else:
            bgr = limit_side(source, self.max_side)
        feat = extract_features(bgr, self._sift)
        return bgr, feat

    # ------------------------------------------------------------------
    # Core pipeline: analyse  (Appendix E verbatim logic)
    # ------------------------------------------------------------------

    def analyse(
        self,
        qfeat: dict,
        exclude_id: str | None = None,
        keep_matches: bool = False,
    ):
        """The whole recognition pipeline for one query (given its SIFT features).
        Returns a plain dictionary of numbers (and optionally per-ref match arrays)."""
        t0 = time.perf_counter()
        per_ref = {}
        for rid in self.ref_ids:
            if rid != exclude_id:
                per_ref[rid] = good_matches(qfeat["desc"], self._feats[rid]["desc"], self.ratio)
        t_match = time.perf_counter() - t0

        rank = landmark_ranking({rid: len(v[0]) for rid, v in per_ref.items()}, self.lm_of)

        t0 = time.perf_counter()
        verified = []
        for lm, rid, n_good in rank[:self.verify_top]:
            qi, ri = per_ref[rid]
            H, mask = ransac_verify(qfeat["pt"], self._feats[rid]["pt"], qi, ri, self.ransac_cfg)
            n_in = int(mask.sum())
            poly, ok, why = None, False, "no homography could be estimated"
            if H is not None and n_in >= 4:
                poly = project_corners(H, self._feats[rid]["shape"])
                ok, why = polygon_check(poly, qfeat["shape"], self.poly_cfg)
            verified.append(dict(
                landmark=lm,
                ref=rid,
                n_good=int(n_good),
                n_inliers=n_in,
                inlier_ratio=(n_in / n_good if n_good else 0.0),
                poly_ok=bool(ok),
                poly_reason=why,
                poly=None if poly is None else np.round(poly, 1).tolist(),
            ))
        verified.sort(key=lambda v: (-v["n_inliers"], -v["n_good"]))
        t_ransac = time.perf_counter() - t0

        res = dict(
            n_kp=int(len(qfeat["pt"])),
            t_sift=float(qfeat["t_sift"]),
            t_match=float(t_match),
            t_ransac=float(t_ransac),
            t_total=float(qfeat["t_sift"] + t_match + t_ransac),
            ranking=[dict(landmark=lm, ref=rid, n_good=int(n)) for lm, rid, n in rank],
            verified=verified,
        )
        return (res, per_ref) if keep_matches else res

    # ------------------------------------------------------------------
    # Public interface: recognize
    # ------------------------------------------------------------------

    def recognize(
        self,
        source: str | Path | np.ndarray,
        thr: dict | None = None,
    ) -> RecognitionResult:
        """Run the full pipeline on a query photo.

        Parameters
        ----------
        source:
            A file path or a BGR image array.
        thr:
            Decision thresholds dict with keys t_good, t_inliers, margin.
            Defaults to the values in landmark_config.json.

        Returns
        -------
        RecognitionResult with all fields populated.
        """
        if thr is None:
            thr = self.thr

        qbgr, qfeat = self._process_query(source)
        res, per_ref = self.analyse(qfeat, keep_matches=True)
        d = decide(res, thr)
        cand = res["verified"][0] if res["verified"] else None
        runner = res["verified"][1] if len(res["verified"]) > 1 else None

        # Build poly message
        poly_msg = "no candidate"
        if cand is not None:
            if cand["poly_ok"] and d["decision"] == "RECOGNIZED":
                poly_msg = "outline drawn (outline checks passed)"
            else:
                reason = cand["poly_reason"] if not cand["poly_ok"] else "the photo was not recognised"
                poly_msg = "outline NOT drawn: " + reason

        # Build report text (Appendix F format)
        source_str = str(source) if isinstance(source, (str, Path)) else "(image array)"
        lines = ["LANDMARK RECOGNITION RESULT"]
        lines.append(f"Query photo: {source_str}")
        lines.append(f"Keypoints found in the query: {res['n_kp']}")
        if cand is not None:
            ref_stem = cand["ref"].split("/")[-1]
            lines.append(f"Most likely landmark: {cand['landmark']} (best reference photo: {ref_stem}.jpg)")
            lines.append(f"Reliable matches (ratio test): {cand['n_good']}")
            lines.append(
                f"RANSAC inliers: {cand['n_inliers']} "
                f"({100 * cand['inlier_ratio']:.0f}% of the reliable matches)"
            )
            if runner is not None:
                lines.append(f"Runner-up: {runner['landmark']} with {runner['n_inliers']} inliers")
        lines.append(f"Localization: {poly_msg}")
        lines.append(
            f"Decision: {d['decision']}"
            + (f" as {d['predicted']}" if d["predicted"] else f" ({d['reason']})")
        )
        lines.append(
            f"Processing time: {res['t_total']:.2f} s "
            f"(SIFT {res['t_sift']:.2f}, matching {res['t_match']:.2f}, "
            f"RANSAC {res['t_ransac']:.2f})"
        )
        report_text = "\n".join(lines)

        # Build images
        img_kp = draw_keypoints(qbgr, qfeat, top=400)

        img_matches: np.ndarray | None = None
        img_outline: np.ndarray | None = None
        ref_bgr: np.ndarray | None = None

        n_raw = 0
        if cand is not None:
            qi_raw, _, _, _ = knn_two(qfeat["desc"], self._feats[cand["ref"]]["desc"])
            n_raw = len(qi_raw)
            qi, ri = per_ref[cand["ref"]]
            H, mask = ransac_verify(
                qfeat["pt"], self._feats[cand["ref"]]["pt"], qi, ri, self.ransac_cfg
            )
            rbgr = load_bgr(self.path_of[cand["ref"]], self.max_side)
            ref_bgr = rbgr
            img_matches = draw_match_lines(
                rbgr, qbgr, self._feats[cand["ref"]]["pt"], qfeat["pt"], ri, qi, mask
            )

            out = qbgr.copy()
            if d["decision"] == "RECOGNIZED" and cand["poly_ok"] and cand["poly"] is not None:
                pts = np.round(np.array(cand["poly"])).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(out, [pts], True, (0, 0, 255), 4, cv2.LINE_AA)
            img_outline = out

        return RecognitionResult(
            decision=d["decision"],
            predicted=d["predicted"],
            candidate=d["candidate"],
            reason=d["reason"],
            count_winner=d["count_winner"],
            n_kp=res["n_kp"],
            t_sift=res["t_sift"],
            t_match=res["t_match"],
            t_ransac=res["t_ransac"],
            t_total=res["t_total"],
            top3=res["verified"],
            report_text=report_text,
            img_keypoints=img_kp,
            img_matches=img_matches,
            img_outline=img_outline,
            ref_bgr=ref_bgr,
            n_raw=n_raw,
        )
