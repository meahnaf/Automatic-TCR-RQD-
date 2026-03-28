#!/usr/bin/env python
# coding: utf-8
# ============================================================================
# TCR / RQD Core Box Analysis – Training Notebook
# ============================================================================
# Run this notebook in Google Colab (GPU runtime recommended).
# File: notebooks/01_TCR_RQD_Training.ipynb
# ============================================================================

# %% [markdown]
# # TCR/RQD Core Box Analysis – Training Pipeline
#
# **Goal**: Train a YOLOv8-seg model to detect and segment individual core
# pieces in borehole core-box photographs, then compute TCR and RQD.
#
# **Classes**
# | ID | Name | Description |
# |----|------|-------------|
# | 0 | `rqd_piece` | Sound core piece >= 10 cm – counts toward both TCR & RQD |
# | 1 | `non_rqd_piece` | Piece < 10 cm or fractured – TCR only |
#
# **Workflow**
# 1. Environment setup
# 2. Data preparation & annotation
# 3. Dataset split
# 4. Model training (YOLOv8n-seg -> YOLOv8s-seg -> YOLOv8m-seg)
# 5. Validation & metrics
# 6. Export weights
# 7. End-to-end inference demo

# %% [markdown]
# ## 1. Environment Setup

# %%
import subprocess, sys

def pip(*args):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *args])

pip('ultralytics>=8.2.0')
pip('roboflow')
pip('openpyxl')
pip('pandas')
pip('matplotlib')

# %%
import os, shutil, json, random, math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from ultralytics import YOLO

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("All dependencies loaded")

# %% [markdown]
# ## 2. Mount Google Drive (optional)

# %%
try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = Path('/content/drive/MyDrive/core_analysis')
except Exception:
    BASE_DIR = Path('/content/core_analysis')

BASE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Working directory: {BASE_DIR}")

# %% [markdown]
# ## 3. Data Organisation
#
# ### Option A – Upload images manually
# Upload your core-box images and YOLO labels to:
# ```
# /content/core_analysis/
# |-- raw_images/        <- your core box photos
# |-- raw_labels/        <- YOLO-format .txt files (same stem as images)
# ```
#
# ### Option B – Import from Roboflow (recommended)

# %%
USE_ROBOFLOW = False  # <- set True and fill in your details

if USE_ROBOFLOW:
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
    project = rf.workspace("YOUR_WORKSPACE").project("core-boxes")
    dataset = project.version(1).download("yolov8")
    DATASET_PATH = Path(dataset.location)
    print(f"Downloaded dataset to {DATASET_PATH}")
else:
    DATASET_PATH = BASE_DIR / 'dataset'
    for split in ['train', 'val', 'test']:
        (DATASET_PATH / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DATASET_PATH / 'labels' / split).mkdir(parents=True, exist_ok=True)
    print(f"Dataset directory created at {DATASET_PATH}")
    print("Copy your images to dataset/images/train and labels to dataset/labels/train")

# %%
yaml_content = f"""
path  : {DATASET_PATH}
train : images/train
val   : images/val
test  : images/test

nc    : 2
names :
  0 : rqd_piece
  1 : non_rqd_piece
"""
yaml_path = DATASET_PATH / 'dataset.yaml'
yaml_path.write_text(yaml_content.strip())
print(f"Wrote {yaml_path}")

# %% [markdown]
# ## 4. Annotation Verification

# %%
CLASS_COLORS = {0: (0, 200, 0), 1: (0, 100, 255)}
CLASS_NAMES  = {0: 'rqd_piece', 1: 'non_rqd_piece'}

def draw_yolo_labels(img_path: Path, lbl_path: Path) -> np.ndarray:
    """Overlay YOLO segmentation polygons on the image."""
    img = cv2.imread(str(img_path))
    if img is None:
        return np.zeros((100, 100, 3), np.uint8)
    h, w = img.shape[:2]
    if not lbl_path.exists():
        return img
    for line in lbl_path.read_text().strip().splitlines():
        parts = list(map(float, line.split()))
        cls = int(parts[0])
        coords = np.array(parts[1:]).reshape(-1, 2)
        coords[:, 0] *= w
        coords[:, 1] *= h
        pts = coords.astype(np.int32)
        color = CLASS_COLORS.get(cls, (128, 128, 128))
        cv2.polylines(img, [pts], True, color, 2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(img, CLASS_NAMES.get(cls, '?'), (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return img

train_imgs = sorted((DATASET_PATH / 'images' / 'train').glob('*.jpg'))[:4]
train_imgs += sorted((DATASET_PATH / 'images' / 'train').glob('*.png'))[:4]
train_imgs = train_imgs[:4]

if train_imgs:
    fig, axes = plt.subplots(1, len(train_imgs), figsize=(5 * len(train_imgs), 5))
    if len(train_imgs) == 1:
        axes = [axes]
    for ax, ip in zip(axes, train_imgs):
        lp = DATASET_PATH / 'labels' / 'train' / (ip.stem + '.txt')
        vis = draw_yolo_labels(ip, lp)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(ip.name, fontsize=8)
        ax.axis('off')
    patches = [mpatches.Patch(color=np.array(v) / 255, label=n)
               for v, n in [(CLASS_COLORS[0], 'rqd_piece'),
                             (CLASS_COLORS[1], 'non_rqd_piece')]]
    fig.legend(handles=patches, loc='lower center', ncol=2)
    plt.tight_layout()
    plt.savefig(str(BASE_DIR / 'annotation_check.png'), dpi=100)
    plt.show()
    print("Annotation check saved")
else:
    print("No training images found – add images to dataset/images/train/")

# %% [markdown]
# ## 5. Dataset Split

# %%
def auto_split_dataset(dataset_path: Path, val_frac=0.15, test_frac=0.05):
    """Split images + labels from train -> val + test."""
    img_dir = dataset_path / 'images' / 'train'
    lbl_dir = dataset_path / 'labels' / 'train'
    images = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
    if len(images) < 5:
        print(f"Only {len(images)} images – skipping split")
        return
    random.shuffle(images)
    n_val = max(1, int(len(images) * val_frac))
    n_test = max(1, int(len(images) * test_frac))
    splits = {
        'val': images[:n_val],
        'test': images[n_val:n_val + n_test],
    }
    for split, imgs in splits.items():
        for ip in imgs:
            dst_img = dataset_path / 'images' / split / ip.name
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ip), str(dst_img))
            lp = lbl_dir / (ip.stem + '.txt')
            if lp.exists():
                dst_lbl = dataset_path / 'labels' / split / lp.name
                dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(lp), str(dst_lbl))
    for split in ['train', 'val', 'test']:
        n = len(list((dataset_path / 'images' / split).glob('*.*')))
        print(f"  {split:6s}: {n} images")

val_images = list((DATASET_PATH / 'images' / 'val').glob('*.*'))
if not val_images:
    print("Running auto-split...")
    auto_split_dataset(DATASET_PATH)
else:
    print(f"Val set already has {len(val_images)} images – skipping split")

# %% [markdown]
# ## 6. Model Training
#
# Progressive training: nano -> small -> medium.

# %%
EXPERIMENT = {
    'model'   : 'yolov8s-seg.pt',
    'epochs'  : 150,
    'imgsz'   : 1024,
    'batch'   : 4,
    'patience': 25,
    'workers' : 2,
    'name'    : 'core_seg_v1',
    'project' : str(BASE_DIR / 'runs'),
    'seed'    : SEED,
    'degrees' : 0,
    'flipud'  : 0.0,
    'fliplr'  : 0.3,
    'hsv_h'   : 0.01,
    'hsv_v'   : 0.30,
    'mosaic'  : 0.6,
    'copy_paste': 0.1,
}

print("Training config:")
for k, v in EXPERIMENT.items():
    print(f"  {k:15s}: {v}")

# %%
model = YOLO(EXPERIMENT.pop('model'))

results = model.train(
    data=str(yaml_path),
    **EXPERIMENT,
    device=0,
    verbose=True,
)

print("\nTraining complete")
best_weights = Path(results.save_dir) / 'weights' / 'best.pt'
print(f"Best weights: {best_weights}")

# %% [markdown]
# ## 7. Validation & Metrics

# %%
model_best = YOLO(str(best_weights))

val_results = model_best.val(
    data=str(yaml_path),
    imgsz=1024,
    split='val',
    device=0,
    verbose=True,
)

# %%
results_csv = Path(results.save_dir) / 'results.csv'
if results_csv.exists():
    df_res = pd.read_csv(results_csv)
    df_res.columns = df_res.columns.str.strip()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    metrics = [
        ('train/box_loss', 'val/box_loss',    'Box Loss'),
        ('train/cls_loss', 'val/cls_loss',    'Class Loss'),
        ('train/seg_loss', 'val/seg_loss',    'Seg Loss'),
        ('metrics/precision(B)', None,        'Precision'),
        ('metrics/recall(B)',    None,        'Recall'),
        ('metrics/mAP50(B)',     None,        'mAP@50'),
    ]
    for ax, (train_col, val_col, title) in zip(axes.flat, metrics):
        if train_col in df_res.columns:
            ax.plot(df_res[train_col], label='train')
        if val_col and val_col in df_res.columns:
            ax.plot(df_res[val_col], label='val')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(BASE_DIR / 'training_curves.png'), dpi=120)
    plt.show()

# %% [markdown]
# ## 8. Export Model

# %%
model_best.export(format='onnx', imgsz=1024, simplify=True)

export_dir = BASE_DIR / 'exported_weights'
export_dir.mkdir(exist_ok=True)
shutil.copy(str(best_weights), str(export_dir / 'best.pt'))
print(f"Weights saved to {export_dir / 'best.pt'}")

# %% [markdown]
# ## Done!
# ---
# **Artefacts produced**
# | File | Description |
# |------|-------------|
# | `runs/core_seg_v1/weights/best.pt` | Best trained weights |
# | `runs/core_seg_v1/results.csv` | Training metrics per epoch |
# | `training_curves.png` | Loss / mAP plots |
# | `annotation_check.png` | Label verification grid |
# | `exported_weights/best.pt` | Copy of best weights for inference |
