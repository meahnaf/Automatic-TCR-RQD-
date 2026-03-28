"""
Test script for metrics module.

Usage:
    python test_metrics.py                    # test all images
    python test_metrics.py data/1.png         # test single image
"""
import sys
import os
import glob
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils.run_segmentation import extract_runs
from utils.piece_detection import detect_pieces
from utils.metrics import compute_lengths, compute_metrics

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

if len(sys.argv) > 1:
    images = [sys.argv[1]]
else:
    images = sorted(glob.glob(os.path.join(DATA_DIR, "*.png")))

if not images:
    print("No PNG images found in", DATA_DIR)
    sys.exit(1)

for img_path in images:
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(img_path)}")
    print(f"{'='*60}")

    img = cv2.imread(img_path)
    if img is None:
        print(f"  ERROR: could not read {img_path}")
        continue

    # --- Extract runs ---
    runs = extract_runs(img)
    print(f"  Detected {len(runs)} runs")

    # --- Process each run ---
    all_tcr = []
    all_rqd = []
    for i, run in enumerate(runs):
        pieces = detect_pieces(run)
        lengths = compute_lengths(pieces, run)
        metrics = compute_metrics(lengths)
        
        print(f"\n  Run {i+1}:")
        print(f"    Pieces: {metrics['n_pieces']} (RQD-eligible: {metrics['n_rqd_pieces']})")
        print(f"    Lengths: {lengths[:3]}{'...' if len(lengths) > 3 else ''}")
        print(f"    Total recovered: {metrics['total_length']} cm")
        print(f"    TCR: {metrics['tcr']}%")
        print(f"    RQD: {metrics['rqd']}%")
        
        all_tcr.append(metrics['tcr'])
        all_rqd.append(metrics['rqd'])

    # --- Summary ---
    if all_tcr:
        print(f"\n  SUMMARY for {os.path.basename(img_path)}:")
        print(f"    Avg TCR: {np.mean(all_tcr):.1f}%  (range: {min(all_tcr):.1f}–{max(all_tcr):.1f}%)")
        print(f"    Avg RQD: {np.mean(all_rqd):.1f}%  (range: {min(all_rqd):.1f}–{max(all_rqd):.1f}%)")
        print(f"    Runs processed: {len(all_tcr)}")

    if len(images) > 1:
        input("\n  Press Enter for next image...")
