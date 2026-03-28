"""
Test script for piece detection module.

Usage:
    python test_piece_detection.py                    # test all images
    python test_piece_detection.py data/1.png         # test single image
"""
import sys
import os
import glob
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils.run_segmentation import extract_runs
from utils.piece_detection import detect_pieces, visualize_pieces

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
    n_runs = len(runs)
    print(f"  Detected {n_runs} runs")

    if n_runs == 0:
        print("  WARNING: no runs detected!")
        continue

    # --- Detect pieces in all runs, show on one figure ---
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(n_runs, 1, figsize=(16, 3 * n_runs), squeeze=False)

    for i, run in enumerate(runs):
        pieces = detect_pieces(run)
        print(f"  Run {i+1}: {len(pieces)} pieces")
        for j, (x, y, w, h) in enumerate(pieces):
            print(f"    Piece {j+1}: x={x} y={y} w={w} h={h}")

        ax = axes[i, 0]
        disp = cv2.cvtColor(run, cv2.COLOR_BGR2RGB)
        ax.imshow(disp)
        for j, (x, y, w, h) in enumerate(pieces):
            rect = mpatches.Rectangle((x, y), w, h,
                                      linewidth=2, edgecolor="lime",
                                      facecolor="none")
            ax.add_patch(rect)
            ax.text(x + w / 2, y - 4, str(j + 1), color="lime",
                    fontsize=8, fontweight="bold", ha="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6))
        rh, rw = run.shape[:2]
        crop_rect = mpatches.Rectangle(
            (int(rw * 0.02), int(rh * 0.10)),
            rw - int(rw * 0.08), rh - int(rh * 0.15),
            linewidth=1, edgecolor="red", facecolor="none", ls="--")
        ax.add_patch(crop_rect)
        ax.set_title(f"Run {i+1}  —  {len(pieces)} pieces", fontsize=10)
        ax.axis("off")

    plt.suptitle(f"{os.path.basename(img_path)}  ({n_runs} runs)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()
