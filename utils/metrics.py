"""
Core Metrics: Piece Lengths, TCR, RQD, and Export
==================================================

Converts pixel-based bounding boxes to real-world lengths and
computes standard geotechnical core metrics.

Definitions
-----------
- **TCR** (Total Core Recovery, %) = total_recovered_length / run_length × 100
- **RQD** (Rock Quality Designation, %) = sum(pieces ≥ 10 cm) / run_length × 100

RQD Quality Classification (Deere 1968)
----------------------------------------
- 90–100 % → Excellent
- 75–90  % → Good
- 50–75  % → Fair
- 25–50  % → Poor
-  0–25  % → Very Poor

Public API
----------
- ``compute_lengths(boxes, run_image, run_length_cm)`` → list of lengths in cm
- ``compute_metrics(lengths_cm, run_length_cm)`` → dict with TCR, RQD, counts
- ``rqd_quality(rqd_pct)`` → quality string
- ``export_results(results, path, fmt)`` → write CSV / JSON / XLSX
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

log = logging.getLogger(__name__)

RUN_LENGTH_CM = 150.0


# ---------------------------------------------------------------------------
# RQD quality classification (Deere 1968)
# ---------------------------------------------------------------------------

def rqd_quality(rqd_pct: float) -> str:
    """Return rock quality string based on RQD percentage."""
    if rqd_pct >= 90:
        return "Excellent"
    if rqd_pct >= 75:
        return "Good"
    if rqd_pct >= 50:
        return "Fair"
    if rqd_pct >= 25:
        return "Poor"
    return "Very Poor"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

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
        - ``rqd_class``     : Deere 1968 quality string.
    """
    total = sum(lengths_cm)
    rqd_lengths = [l for l in lengths_cm if l >= rqd_min_cm]
    rqd_total = sum(rqd_lengths)

    tcr = min(100.0, (total / run_length_cm) * 100.0) if run_length_cm > 0 else 0.0
    rqd = min(100.0, (rqd_total / run_length_cm) * 100.0) if run_length_cm > 0 else 0.0

    return {
        "tcr": round(tcr, 2),
        "rqd": round(rqd, 2),
        "total_length": round(total, 2),
        "rqd_length": round(rqd_total, 2),
        "n_pieces": len(lengths_cm),
        "n_rqd_pieces": len(rqd_lengths),
        "run_length_cm": run_length_cm,
        "rqd_class": rqd_quality(rqd),
    }


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_results(
    results: dict,
    out_path: str | Path,
    fmt: str = "json",
) -> Path:
    """Export pipeline results to file.

    Parameters
    ----------
    results : dict as produced by the UI pipeline (with 'runs', 'summary', etc.)
    out_path : output file path (extension overridden by *fmt*).
    fmt : 'json', 'csv', or 'xlsx'.

    Returns
    -------
    Path to the written file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        out_path = out_path.with_suffix(".json")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    elif fmt == "csv":
        out_path = out_path.with_suffix(".csv")
        _write_csv(results, out_path)

    elif fmt == "xlsx":
        out_path = out_path.with_suffix(".xlsx")
        _write_xlsx(results, out_path)

    log.info("Exported %s → %s", fmt.upper(), out_path)
    return out_path


def _results_to_rows(results: dict) -> list[dict]:
    """Flatten results dict into a list of per-run row dicts."""
    rows = []
    for r in results.get("runs", []):
        m = r.get("metrics", r)
        rows.append({
            "Run": r.get("run", ""),
            "TCR (%)": m.get("tcr", ""),
            "RQD (%)": m.get("rqd", ""),
            "TCR (cm)": m.get("total_length", ""),
            "RQD (cm)": m.get("rqd_length", ""),
            "Rock Quality": m.get("rqd_class", ""),
            "# Pieces": m.get("n_pieces", ""),
            "# RQD Pieces": m.get("n_rqd_pieces", ""),
        })
    return rows


def _write_csv(results: dict, path: Path) -> None:
    """Write results as CSV (no pandas required)."""
    rows = _results_to_rows(results)
    if not rows:
        path.write_text("")
        return
    header = list(rows[0].keys())
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row[k]) for k in header))
    path.write_text("\n".join(lines))


def _write_xlsx(results: dict, path: Path) -> None:
    """Write results as XLSX (requires openpyxl)."""
    try:
        import pandas as pd
        rows = _results_to_rows(results)
        df = pd.DataFrame(rows)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="TCR_RQD", index=False)
            ws = writer.sheets["TCR_RQD"]
            for col in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in col)
                ws.column_dimensions[col[0].column_letter].width = max_len + 3
    except ImportError:
        log.warning("openpyxl/pandas not installed – XLSX export skipped.")
