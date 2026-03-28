"""
Core Metrics: Piece Lengths, TCR, and RQD
==========================================

Converts pixel-based bounding boxes to real-world lengths and
computes standard geotechnical core metrics.

Definitions
-----------
- **TCR** (Total Core Recovery, %) = total_recovered_length / run_length × 100
- **RQD** (Rock Quality Designation, %) = sum(pieces ≥ 10 cm) / run_length × 100

Public API
----------
- ``compute_lengths(boxes, run_image, run_length_cm)`` → list of lengths in cm
- ``compute_metrics(lengths_cm, run_length_cm)`` → dict with TCR, RQD, counts
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def compute_lengths(
    boxes: List[Tuple[int, int, int, int]],
    run_image: np.ndarray,
    run_length_cm: float = 150.0,
) -> List[float]:
    """Convert piece bounding-box widths from pixels to centimetres.

    Parameters
    ----------
    boxes : list of (x, y, w, h) bounding boxes from ``detect_pieces``.
    run_image : the run image (used to get its pixel width for scaling).
    run_length_cm : physical length of the run in cm (default 150 cm = 1.5 m).

    Returns
    -------
    List[float] — piece lengths in cm, same order as *boxes*.
    """
    if not boxes or run_image is None or run_image.size == 0:
        return []

    img_width_px = run_image.shape[1]
    px_to_cm = run_length_cm / img_width_px

    raw = [w * px_to_cm for (_, _, w, _) in boxes]

    # Filter tiny noise pieces (< 5 cm) and cap unrealistic ones
    return [min(l, run_length_cm) for l in raw if l > 5.0]


def compute_metrics(
    lengths_cm: List[float],
    run_length_cm: float = 150.0,
    rqd_min_cm: float = 10.0,
) -> Dict[str, float]:
    """Compute TCR and RQD from a list of piece lengths.

    Parameters
    ----------
    lengths_cm : piece lengths in cm (from ``compute_lengths``).
    run_length_cm : nominal run length in cm (default 150).
    rqd_min_cm : minimum piece length for RQD eligibility (default 10 cm).

    Returns
    -------
    dict with keys:
        - ``tcr``           : TCR as a percentage (0–100+).
        - ``rqd``           : RQD as a percentage (0–100+).
        - ``total_length``  : sum of all piece lengths (cm).
        - ``rqd_length``    : sum of RQD-eligible piece lengths (cm).
        - ``n_pieces``      : total number of pieces.
        - ``n_rqd_pieces``  : number of RQD-eligible pieces.
        - ``run_length_cm`` : the nominal run length used.
    """
    total = sum(lengths_cm)
    rqd_lengths = [l for l in lengths_cm if l >= rqd_min_cm]
    rqd_total = sum(rqd_lengths)

    tcr = (total / run_length_cm) * 100.0 if run_length_cm > 0 else 0.0
    rqd = (rqd_total / run_length_cm) * 100.0 if run_length_cm > 0 else 0.0

    return {
        "tcr": round(tcr, 2),
        "rqd": round(rqd, 2),
        "total_length": round(total, 2),
        "rqd_length": round(rqd_total, 2),
        "n_pieces": len(lengths_cm),
        "n_rqd_pieces": len(rqd_lengths),
        "run_length_cm": run_length_cm,
    }
