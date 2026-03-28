"""
pipeline.py — End-to-end TCR / RQD CLI
========================================

Usage:
    python pipeline.py --image data/1.png
    python pipeline.py --image data/1.png data/2.png --out results/
    python pipeline.py --image data/*.png --fmt csv json
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from utils.run_segmentation import extract_runs
from utils.piece_detection import detect_pieces
from utils.metrics import (
    compute_lengths, compute_metrics, rqd_quality, export_results,
)
from utils.scale_detector import detect_scale

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def process_image(
    image_path: str | Path,
    run_length_cm: float = 150.0,
    use_scale_detection: bool = True,
) -> dict:
    """Process one core-box image through the full pipeline.

    Returns
    -------
    dict with keys: source, run_length_cm, n_runs, summary, runs, scale
    """
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    log.info("Processing %s  (%d×%d px)", image_path.name, *image.shape[:2][::-1])

    # Scale detection
    scale_info = {"px_per_cm": 0.0, "method": "fixed", "confidence": 0.0}
    if use_scale_detection:
        scale_info = detect_scale(image)
        log.info("Scale: %.2f px/cm  [%s]  conf=%.2f",
                 scale_info["px_per_cm"], scale_info["method"],
                 scale_info["confidence"])

    # Run segmentation
    runs = extract_runs(image)
    log.info("Detected %d runs", len(runs))

    # Per-run piece detection + metrics
    run_results = []
    for i, run_img in enumerate(runs):
        boxes = detect_pieces(run_img)
        lengths = compute_lengths(boxes, run_img, run_length_cm)
        metrics = compute_metrics(lengths, run_length_cm)
        run_results.append({
            "run": i + 1,
            "lengths_cm": [round(l, 2) for l in lengths],
            "metrics": metrics,
        })

    # Summary
    tcr_vals = [r["metrics"]["tcr"] for r in run_results]
    rqd_vals = [r["metrics"]["rqd"] for r in run_results]
    total_pieces = sum(r["metrics"]["n_pieces"] for r in run_results)
    avg_tcr = float(np.mean(tcr_vals)) if tcr_vals else 0.0
    avg_rqd = float(np.mean(rqd_vals)) if rqd_vals else 0.0

    result = {
        "source": image_path.name,
        "run_length_cm": run_length_cm,
        "n_runs": len(runs),
        "scale": {
            "px_per_cm": round(scale_info["px_per_cm"], 2),
            "method": scale_info["method"],
            "confidence": round(scale_info["confidence"], 2),
        },
        "summary": {
            "avg_tcr": round(avg_tcr, 2),
            "avg_rqd": round(avg_rqd, 2),
            "total_pieces": total_pieces,
            "rqd_class": rqd_quality(avg_rqd),
        },
        "runs": run_results,
    }

    # Print summary table
    _print_summary(result)
    return result


def _print_summary(result: dict) -> None:
    """Pretty-print results to console."""
    print(f"\n{'='*62}")
    print(f"  Image  : {result['source']}")
    print(f"  Scale  : {result['scale']['px_per_cm']:.2f} px/cm  [{result['scale']['method']}]")
    print(f"{'='*62}")
    print(f"  {'Run':>4}  {'TCR%':>7}  {'RQD%':>7}  {'Quality':<12}  {'Pieces':>6}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*12}  {'-'*6}")
    for r in result["runs"]:
        m = r["metrics"]
        print(f"  {r['run']:>4}  {m['tcr']:>7.1f}  {m['rqd']:>7.1f}  "
              f"{m['rqd_class']:<12}  {m['n_pieces']:>6}")
    s = result["summary"]
    print(f"  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*12}  {'-'*6}")
    print(f"  {'AVG':>4}  {s['avg_tcr']:>7.1f}  {s['avg_rqd']:>7.1f}  "
          f"{s['rqd_class']:<12}  {s['total_pieces']:>6}")
    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="TCR/RQD pipeline for core-box images")
    p.add_argument("--image", required=True, nargs="+", help="Input image(s)")
    p.add_argument("--out", default="results", help="Output directory")
    p.add_argument("--run-length", type=float, default=150.0, help="Run length in cm")
    p.add_argument("--fmt", nargs="+", default=["json"], choices=["json", "csv", "xlsx"],
                   help="Export format(s)")
    p.add_argument("--no-scale", action="store_true", help="Skip scale bar detection")
    return p.parse_args()


def main():
    args = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in args.image:
        try:
            result = process_image(
                img_path,
                run_length_cm=args.run_length,
                use_scale_detection=not args.no_scale,
            )
            stem = Path(img_path).stem
            for fmt in args.fmt:
                export_results(result, out_dir / f"{stem}_tcr_rqd", fmt)
        except Exception as exc:
            log.error("Failed to process %s: %s", img_path, exc)


if __name__ == "__main__":
    main()
