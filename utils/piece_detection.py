"""
Piece Detection for Borehole Core Run Images
=============================================

Strategy: **Otsu binary + Canny edge → combined column projection → slicing**

1.  Crop margins (top labels, left/right tray frame, right depth markers).
2.  Convert to grayscale, apply Gaussian blur.
3.  Otsu binary threshold → rock mask (rock white, tray/gaps black).
4.  Canny edge detection → crack/boundary signal.
5.  Column-wise projection of rock mask − edge penalty → piece score.
6.  Smooth lightly, Otsu-threshold the 1-D score.
7.  Contiguous bright regions → piece column ranges.
8.  For each piece, find vertical extent → bounding box.
9.  Filter tiny pieces (noise).

Public API
----------
- ``detect_pieces(run_image, ...)`` → ``List[Tuple[int, int, int, int]]``
- ``visualize_pieces(run_image, ...)`` → matplotlib figure
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


def _crop_margins(
    image: np.ndarray,
    top: float = 0.10,
    bottom: float = 0.05,
    left: float = 0.02,
    right: float = 0.06,
) -> Tuple[np.ndarray, int, int]:
    """Crop margins and return (cropped, y_offset, x_offset)."""
    h, w = image.shape[:2]
    y1 = int(h * top)
    y2 = h - int(h * bottom)
    x1 = int(w * left)
    x2 = w - int(w * right)
    return image[y1:y2, x1:x2], y1, x1


def _smooth_1d(signal: np.ndarray, smooth_frac: float) -> np.ndarray:
    """Moving-average smoothing of a 1-D signal."""
    n = len(signal)
    k = max(3, int(n * smooth_frac))
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k) / k
    return np.convolve(signal, kernel, mode="same")


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


def _find_bright_regions(
    signal: np.ndarray, threshold: float,
) -> List[Tuple[int, int]]:
    """Contiguous column-ranges where signal > threshold (bright = rock)."""
    above = signal > threshold
    regions: List[Tuple[int, int]] = []
    in_region = False
    start = 0
    for i, v in enumerate(above):
        if v and not in_region:
            in_region, start = True, i
        elif not v and in_region:
            in_region = False
            regions.append((start, i))
    if in_region:
        regions.append((start, len(signal)))
    return regions


def _vertical_extent(
    gray: np.ndarray, x1: int, x2: int, bright_frac: float = 0.25,
) -> Tuple[int, int]:
    """Find top/bottom rows with significant bright content in [x1:x2]."""
    col_slice = gray[:, x1:x2]
    row_mean = col_slice.mean(axis=1)
    thr = row_mean.max() * bright_frac
    bright = np.where(row_mean > thr)[0]
    if len(bright) == 0:
        return 0, gray.shape[0]
    return int(bright[0]), int(bright[-1] + 1)


def _piece_score(blur: np.ndarray, edge_weight: float = 0.4) -> np.ndarray:
    """Combined column-wise piece score: binary rock projection − edge penalty.

    Otsu binary separates rock (white) from tray/gaps (black),
    giving a clean column-wise rock-presence signal.
    Canny edges add a penalty at crack/boundary locations so
    narrow cracks that survive binarisation still get detected.
    """
    # Rock mask (Otsu) — rock = 255, tray/gaps = 0
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Canny edges — strong at cracks and piece boundaries
    edges = cv2.Canny(blur, 50, 150)

    # Column-wise projections, normalised to 0-1
    rock_proj = binary.astype(np.float64).mean(axis=0) / 255.0
    edge_proj = edges.astype(np.float64).mean(axis=0) / 255.0

    # Combined: high rock presence, penalise crack columns
    return rock_proj - edge_weight * edge_proj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_pieces(
    run_image: np.ndarray,
    *,
    crop_top: float = 0.10,
    crop_bottom: float = 0.05,
    crop_left: float = 0.02,
    crop_right: float = 0.06,
    blur_ksize: int = 5,
    smooth_frac: float = 0.015,
    edge_weight: float = 0.4,
    min_piece_width: int = 25,
) -> List[Tuple[int, int, int, int]]:
    """Detect horizontal core pieces in a run image.

    Parameters
    ----------
    run_image : BGR / RGB / grayscale run photograph.
    crop_top/bottom/left/right : margin fractions to crop.
    smooth_frac : smoothing kernel as fraction of image width.
    edge_weight : how much Canny edges penalise the piece score.
    min_piece_width : minimum piece width in pixels.

    Returns
    -------
    List[Tuple[int, int, int, int]]  — bounding boxes (x, y, w, h) in
    the original (uncropped) image coordinate system.
    """
    if run_image is None or run_image.size == 0:
        return []

    # ---- 1. crop margins ----
    cropped, y_off, x_off = _crop_margins(
        run_image, top=crop_top, bottom=crop_bottom,
        left=crop_left, right=crop_right,
    )

    # ---- 2. grayscale + blur ----
    gray = _to_gray(cropped)
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # ---- 3. combined piece score (binary rock − edge penalty) ----
    score = _piece_score(blur, edge_weight)

    # ---- 4. smooth lightly ----
    smooth = _smooth_1d(score, smooth_frac)

    # ---- 5. Otsu on the 1-D score ----
    thr = _otsu_1d(smooth)
    thr = max(thr, 0.3)  # floor so we don't merge everything

    # ---- 6. bright regions = pieces ----
    regions = _find_bright_regions(smooth, thr)

    if not regions:
        return []

    # ---- 7. bounding boxes ----
    boxes: List[Tuple[int, int, int, int]] = []
    for rx1, rx2 in regions:
        pw = rx2 - rx1
        if pw < min_piece_width:
            continue
        ry1, ry2 = _vertical_extent(gray, rx1, rx2)
        # map back to original image coords
        x = rx1 + x_off
        y = ry1 + y_off
        w = pw
        h = ry2 - ry1
        if h < 5:
            continue
        boxes.append((x, y, w, h))

    return boxes


def visualize_pieces(
    run_image: np.ndarray,
    *,
    crop_top: float = 0.10,
    crop_bottom: float = 0.05,
    crop_left: float = 0.02,
    crop_right: float = 0.06,
    blur_ksize: int = 5,
    smooth_frac: float = 0.015,
    edge_weight: float = 0.4,
) -> None:
    """Debug visualisation for piece detection."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if run_image is None or run_image.size == 0:
        return

    boxes = detect_pieces(
        run_image,
        crop_top=crop_top, crop_bottom=crop_bottom,
        crop_left=crop_left, crop_right=crop_right,
        blur_ksize=blur_ksize, smooth_frac=smooth_frac,
        edge_weight=edge_weight,
    )

    # Recompute internals for plotting
    cropped, y_off, x_off = _crop_margins(
        run_image, top=crop_top, bottom=crop_bottom,
        left=crop_left, right=crop_right,
    )
    gray = _to_gray(cropped)
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    score = _piece_score(blur, edge_weight)
    smooth = _smooth_1d(score, smooth_frac)
    thr = max(_otsu_1d(smooth), 0.3)

    # Also get the binary mask for display
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    disp = cv2.cvtColor(run_image, cv2.COLOR_BGR2RGB) if len(run_image.shape) == 3 \
        else cv2.cvtColor(run_image, cv2.COLOR_GRAY2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5),
                             gridspec_kw={"width_ratios": [3, 1, 1]})

    # Left: image + boxes + crop rectangle
    axes[0].imshow(disp)
    rh, rw = run_image.shape[:2]
    crop_rect = mpatches.Rectangle(
        (int(rw * crop_left), int(rh * crop_top)),
        rw - int(rw * (crop_left + crop_right)),
        rh - int(rh * (crop_top + crop_bottom)),
        linewidth=1, edgecolor="red", facecolor="none", ls="--",
    )
    axes[0].add_patch(crop_rect)
    for j, (x, y, w, h) in enumerate(boxes):
        rect = mpatches.Rectangle((x, y), w, h,
                                  linewidth=2, edgecolor="lime",
                                  facecolor="none")
        axes[0].add_patch(rect)
        axes[0].text(x + w / 2, y - 3, str(j + 1), color="lime",
                     fontsize=8, fontweight="bold", ha="center",
                     bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6))
    axes[0].set_title(f"{len(boxes)} pieces detected")
    axes[0].axis("off")

    # Middle: piece score signal + threshold
    cols = np.arange(len(smooth))
    axes[1].plot(cols, smooth, lw=0.8, color="steelblue", label="score")
    axes[1].plot(cols, score, lw=0.3, color="gray", alpha=0.4, label="raw")
    axes[1].axhline(thr, color="red", ls="--", lw=1, label=f"thr={thr:.2f}")
    axes[1].set_title("Piece score (rock − edges)")
    axes[1].set_xlabel("Column")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    # Right: binary rock mask
    axes[2].imshow(binary, cmap="gray", aspect="auto")
    axes[2].set_title("Otsu binary (rock = white)")

    plt.tight_layout()
    plt.show()
