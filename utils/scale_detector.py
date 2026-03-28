"""
Scale Bar Detection for Core-Box Images
========================================

Detects the ruler / scale bar printed at the top of every core-box image
and returns a pixel-to-centimetre conversion factor (px/cm).

Strategy (ordered by reliability):
1. Alternating-block pattern detection in top strip
2. Tick-line spacing via edge projection
3. Fallback: assume image width ≈ 120 cm

Public API
----------
- ``detect_scale(image, ...)`` → dict with ``px_per_cm``, ``method``, ``confidence``
"""

from __future__ import annotations
import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_scale(image: np.ndarray, debug: bool = False) -> dict:
    """Analyse the scale bar region at the top of *image*.

    Returns
    -------
    dict with keys
        px_per_cm   : float
        method      : str   – which algorithm succeeded
        confidence  : float – 0-1
        debug_image : np.ndarray | None
    """
    h, w = image.shape[:2]

    # Isolate scale-bar strip (top ~8 %)
    strip_h = max(30, int(h * 0.08))
    strip = image[:strip_h, :]

    result = _try_alternating_blocks(strip, w)
    if result is None:
        result = _try_tick_spacing(strip, w)
    if result is None:
        result = _fallback_estimate(w)

    if debug:
        dbg = image.copy()
        cv2.rectangle(dbg, (0, 0), (w, strip_h), (0, 255, 255), 2)
        cv2.putText(dbg,
                    f"{result['px_per_cm']:.2f} px/cm  [{result['method']}]",
                    (10, strip_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        result['debug_image'] = dbg

    log.debug("Scale: %.2f px/cm via '%s' (conf=%.2f)",
              result['px_per_cm'], result['method'], result['confidence'])
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_alternating_blocks(strip: np.ndarray, full_w: int) -> dict | None:
    """Detect alternating black/white 10-cm blocks in the ruler strip."""
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if len(strip.shape) == 3 else strip

    # Average each column vertically (top half only to avoid label text)
    col_avg = gray[:gray.shape[0] // 2, :].mean(axis=0)

    # Smooth and binarise
    col_smooth = np.convolve(col_avg, np.ones(5) / 5, mode='same')
    thresh = (col_smooth.max() + col_smooth.min()) / 2
    binary = (col_smooth < thresh).astype(np.uint8)

    # Find transitions
    transitions = np.where(np.diff(binary.astype(int)) != 0)[0]
    if len(transitions) < 4:
        return None

    # Block widths
    widths = np.diff(transitions)
    widths = widths[(widths > 10) & (widths < 500)]
    if len(widths) < 3:
        return None

    mean_block_px = float(np.median(widths))
    px_per_cm = mean_block_px / 10.0  # each block = 10 cm
    confidence = min(1.0, len(widths) / 10.0)

    return {
        'px_per_cm': px_per_cm,
        'method': 'alternating_blocks',
        'confidence': confidence,
        'debug_image': None,
    }


def _try_tick_spacing(strip: np.ndarray, full_w: int) -> dict | None:
    """Find tick lines via Canny edge projection and measure spacing."""
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if len(strip.shape) == 3 else strip
    edges = cv2.Canny(gray, 30, 100)

    proj = edges.sum(axis=0).astype(float)
    proj /= (proj.max() + 1e-9)

    # Find peaks
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(proj, height=0.1, distance=15)
    except ImportError:
        # Pure-numpy fallback: local maxima above threshold
        threshold = 0.2
        above = proj > threshold
        diff_l = np.r_[True, proj[1:] >= proj[:-1]]
        diff_r = np.r_[proj[:-1] >= proj[1:], True]
        peaks = np.where(above & diff_l & diff_r)[0]

    if len(peaks) < 3:
        return None

    spacings = np.diff(peaks)
    spacings = spacings[(spacings > 10) & (spacings < 500)]
    if len(spacings) < 2:
        return None

    mean_spacing = float(np.median(spacings))
    px_per_cm = mean_spacing / 10.0

    return {
        'px_per_cm': px_per_cm,
        'method': 'tick_spacing',
        'confidence': 0.7,
        'debug_image': None,
    }


def _fallback_estimate(full_w: int) -> dict:
    """Assume image width ≈ 120 cm (standard core-box photograph)."""
    px_per_cm = full_w / 120.0
    log.warning("Scale bar not detected – fallback estimate %.2f px/cm", px_per_cm)
    return {
        'px_per_cm': px_per_cm,
        'method': 'fallback_120cm',
        'confidence': 0.3,
        'debug_image': None,
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/1.png'
    img = cv2.imread(path)
    if img is None:
        print(f"Cannot read {path}")
        sys.exit(1)
    res = detect_scale(img, debug=True)
    print(f"px/cm = {res['px_per_cm']:.3f}  method = {res['method']}  "
          f"confidence = {res['confidence']:.2f}")
