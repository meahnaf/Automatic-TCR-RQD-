"""
Dynamic Run Segmentation for Borehole Core Box Images
======================================================

Strategy: **Smoothed horizontal projection → valley detection → slicing**

1.  Convert to grayscale, apply light Gaussian blur.
2.  Compute row-wise mean intensity (horizontal projection).
3.  Smooth the 1-D projection with a large moving-average kernel so
    each run collapses into one bright block and separators become
    dark valleys.
4.  Otsu-threshold the smoothed 1-D signal to separate bright (core)
    rows from dark (separator) rows.
5.  Contiguous dark rows → separator bands.
6.  Clean: drop tiny bands, merge close bands.
7.  Slice directly between consecutive band edges.
8.  Optionally discard empty runs.

Public API
----------
- ``extract_runs(image, ...)`` → ``List[np.ndarray]``
- ``visualize_projection(image, ...)`` → matplotlib figure
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _smooth_projection(gray: np.ndarray, smooth_frac: float = 0.03) -> np.ndarray:
    """Row-wise mean intensity, smoothed with a 1-D moving average.

    Heavy smoothing collapses individual core pieces within each run
    into one bright block.  Metal separators remain as dark valleys.
    """
    h, w = gray.shape
    projection = gray.mean(axis=1).astype(np.float64)
    k = max(3, int(h * smooth_frac))
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k) / k
    return np.convolve(projection, kernel, mode="same")


def _otsu_1d(signal: np.ndarray) -> float:
    """Otsu threshold for a 1-D signal, returned in the signal's scale."""
    smin, smax = float(signal.min()), float(signal.max())
    span = smax - smin
    if span < 1e-6:
        return smin
    norm = ((signal - smin) / span * 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(
        norm.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return smin + (float(otsu_val) / 255.0) * span


def _find_dark_bands(
    signal: np.ndarray, threshold: float,
) -> List[Tuple[int, int]]:
    """Contiguous row-ranges where signal < threshold (dark valleys)."""
    below = signal < threshold
    bands: List[Tuple[int, int]] = []
    in_band = False
    start = 0
    for i, v in enumerate(below):
        if v and not in_band:
            in_band, start = True, i
        elif not v and in_band:
            in_band = False
            bands.append((start, i))
    if in_band:
        bands.append((start, len(signal)))
    return bands


def _clean_bands(
    bands: List[Tuple[int, int]],
    min_gap: int,
    min_h: int = 2,
) -> List[Tuple[int, int]]:
    """Drop tiny noise bands and merge close ones."""
    bands = [(s, e) for s, e in bands if e - s >= min_h]
    if not bands:
        return bands
    merged = [bands[0]]
    for s, e in bands[1:]:
        if s - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def _is_empty_run(
    gray: np.ndarray, bright_ratio: float = 0.12, bright_val: int = 140,
) -> bool:
    if gray.size == 0:
        return True
    return float(np.sum(gray > bright_val)) / gray.size < bright_ratio


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_runs(
    image: np.ndarray,
    *,
    blur_ksize: int = 5,
    smooth_frac: float = 0.04,
    min_run_height_frac: float = 0.035,
    remove_empty: bool = True,
    empty_bright_ratio: float = 0.12,
) -> List[np.ndarray]:
    """Segment a core-box image into individual horizontal run strips.

    Parameters
    ----------
    image : BGR / RGB / grayscale core-box photograph.
    smooth_frac : smoothing kernel size as fraction of image height.
    min_run_height_frac : minimum run height as fraction of image height.
    remove_empty : discard nearly-empty trays.
    empty_bright_ratio : bright-pixel ratio below which a run is empty.

    Returns
    -------
    List[np.ndarray]  — cropped run images, top to bottom.
    """
    if image is None or image.size == 0:
        return []

    h, w = image.shape[:2]
    gray = _to_gray(image)
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    min_run_h = int(min_run_height_frac * h)

    # ---- 1. smoothed horizontal projection ----
    proj = _smooth_projection(blur, smooth_frac)

    # ---- 2. Otsu threshold on the 1-D projection ----
    thr = _otsu_1d(proj)

    # ---- 3. find & clean separator bands (dark valleys) ----
    raw_bands = _find_dark_bands(proj, thr)
    bands = _clean_bands(raw_bands, min_gap=max(3, h // 60))

    if len(bands) < 2:
        return [image]

    # ---- 4. slice directly between band edges ----
    runs: List[np.ndarray] = []
    for i in range(len(bands) - 1):
        y1 = bands[i][1]       # end of dark valley i
        y2 = bands[i + 1][0]   # start of dark valley i+1
        if y2 - y1 >= min_run_h:
            runs.append(image[y1:y2, :])

    # ---- 5. remove empty runs ----
    if remove_empty and runs:
        runs = [r for r in runs
                if not _is_empty_run(_to_gray(r), bright_ratio=empty_bright_ratio)]

    return runs


def visualize_projection(
    image: np.ndarray,
    *,
    blur_ksize: int = 5,
    smooth_frac: float = 0.04,
) -> None:
    """Debug visualisation: smoothed projection + detected valleys + runs."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    h, w = image.shape[:2]
    gray = _to_gray(image)
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    min_run_h = int(0.035 * h)

    proj = _smooth_projection(blur, smooth_frac)
    thr = _otsu_1d(proj)
    raw_bands = _find_dark_bands(proj, thr)
    bands = _clean_bands(raw_bands, min_gap=max(3, h // 60))

    # Build run slices
    run_slices: List[Tuple[int, int]] = []
    for i in range(len(bands) - 1):
        y1, y2 = bands[i][1], bands[i + 1][0]
        if y2 - y1 >= min_run_h:
            run_slices.append((y1, y2))

    n_bands = len(bands)
    n_runs = len(run_slices)

    disp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 \
        else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(20, 10),
                             gridspec_kw={"width_ratios": [2, 1, 1]})

    # Left: image + dark-valley bands (red) + run edges (green)
    axes[0].imshow(disp)
    for s, e in bands:
        rect = mpatches.Rectangle((0, s), w, e - s,
                                  linewidth=0, edgecolor="none",
                                  facecolor="red", alpha=0.35)
        axes[0].add_patch(rect)
    for i, (y1, y2) in enumerate(run_slices):
        axes[0].axhline(y1, color="lime", lw=1.2, ls="--")
        axes[0].axhline(y2, color="lime", lw=1.2, ls="--")
        axes[0].text(5, (y1 + y2) // 2, f"Run {i+1}", color="lime",
                     fontsize=9, fontweight="bold", va="center",
                     bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))
    axes[0].set_title(f"{n_bands} valleys \u2192 {n_runs} runs  (thr={thr:.1f})")
    axes[0].set_ylabel("Row")

    # Middle: smoothed projection + threshold
    rows = np.arange(h)
    axes[1].plot(proj, rows, lw=0.8, color="steelblue", label="projection")
    axes[1].axvline(thr, color="red", ls="--", label=f"Otsu={thr:.1f}")
    for s, e in bands:
        axes[1].axhspan(s, e, alpha=0.15, color="red")
    axes[1].set_ylim(h, 0)
    axes[1].set_title("Smoothed horizontal projection")
    axes[1].legend(fontsize=8)

    # Right: heavily blurred grayscale (shows the "block" effect)
    ky = max(3, int(h * smooth_frac * 3))
    if ky % 2 == 0:
        ky += 1
    heavy = cv2.GaussianBlur(gray, (1, ky), 0)
    axes[2].imshow(heavy, cmap="gray", aspect="auto")
    axes[2].set_title("Vertically smoothed grayscale")

    plt.tight_layout()
    plt.show()
