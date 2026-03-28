"""
Quick test script for the run segmentation module.
Run from the project root:  python test_segmentation.py

Pass a single filename to test one image:
    python test_segmentation.py data/1.png
Or run with no args to test all PNGs in data/.
"""
import sys
import os
import glob
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils.run_segmentation import extract_runs, visualize_projection

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

    # --- Show projection debug ---
    visualize_projection(img)

    # --- Extract runs ---
    runs = extract_runs(img)
    n = len(runs)
    print(f"  Detected {n} runs")

    if n == 0:
        print("  WARNING: no runs detected!")
        continue

    # --- Display each run ---
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n))
    if n == 1:
        axes = [axes]
    for i, run in enumerate(runs):
        disp = cv2.cvtColor(run, cv2.COLOR_BGR2RGB)
        axes[i].imshow(disp)
        axes[i].set_title(f"Run {i+1}  ({run.shape[0]}×{run.shape[1]})")
        axes[i].axis("off")
    plt.suptitle(os.path.basename(img_path), fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()
