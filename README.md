# TCR / RQD Automatic Calculator

A production-ready computer vision pipeline for automatically calculating **Total Core Recovery (TCR)** and **Rock Quality Designation (RQD)** from borehole core-box photographs.

---

## Features

- **Scale Bar Detection** — automatic px/cm conversion from alternating-block rulers
- **Run Segmentation** — detects horizontal core runs via projection-based Otsu thresholding
- **Piece Detection** — identifies individual core pieces using Otsu binary + Canny edge detection
- **TCR / RQD Calculation** — with RQD quality classification (Deere 1968)
- **Web UI** — Streamlit app: multi-image upload, progress bar, colour-coded quality badges
- **CLI** — batch processing from the command line
- **Multi-Format Export** — JSON, CSV, XLSX
- **Training Notebook** — Google Colab notebook for YOLOv8-seg training

---

## Definitions

| Metric | Formula |
|--------|---------|
| **TCR** | (sum of all recovered piece lengths / 150 cm) × 100 |
| **RQD** | (sum of sound pieces ≥ 10 cm / 150 cm) × 100 |

**RQD Quality Classification (Deere 1968)**

| RQD % | Quality |
|--------|---------|
| 90–100 | Excellent |
| 75–90  | Good |
| 50–75  | Fair |
| 25–50  | Poor |
| 0–25   | Very Poor |

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Web UI

```bash
streamlit run ui/app.py
```

### CLI

```bash
# Single image
python pipeline.py --image data/1.png

# Multiple images with CSV + JSON export
python pipeline.py --image data/1.png data/2.png --out results/ --fmt csv json

# Skip scale bar detection (use fixed 150 cm)
python pipeline.py --image data/1.png --no-scale
```

### Python API

```python
import cv2
from utils.run_segmentation import extract_runs
from utils.piece_detection import detect_pieces
from utils.metrics import compute_lengths, compute_metrics
from utils.scale_detector import detect_scale

image = cv2.imread('data/1.png')

# Optional: detect scale bar
scale = detect_scale(image)
print(f"Scale: {scale['px_per_cm']:.2f} px/cm [{scale['method']}]")

# Pipeline
runs = extract_runs(image)
for i, run in enumerate(runs):
    boxes = detect_pieces(run)
    lengths = compute_lengths(boxes, run, run_length_cm=150.0)
    metrics = compute_metrics(lengths, run_length_cm=150.0)
    print(f"Run {i+1}: TCR={metrics['tcr']:.1f}%  RQD={metrics['rqd']:.1f}%  "
          f"Quality={metrics['rqd_class']}")
```

---

## Pipeline Architecture

```
Core Box Image
      │
      ▼
┌─────────────────────┐
│  scale_detector.py  │  alternating-block pattern → px/cm
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ run_segmentation.py │  row-wise projection + Otsu → run bands
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ piece_detection.py  │  Otsu binary + Canny edges → piece boxes
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│     metrics.py      │  TCR, RQD, quality classification
└─────────┬───────────┘
          │
          ▼
   JSON / CSV / XLSX
   Streamlit UI
   CLI output
```

---

## Algorithm Details

### Scale Bar Detection
1. Isolate top ~8% of image (ruler strip)
2. Detect alternating black/white blocks → median block width
3. Fallback: tick-line spacing via Canny edge projection
4. Last resort: assume image width ≈ 120 cm

### Run Segmentation
1. Convert to grayscale + Gaussian blur
2. Row-wise mean projection (horizontal signal)
3. Otsu threshold on 1D projection
4. Split at dark valleys (metal separator bands)

### Piece Detection
1. Crop margins (top labels, tray edges, depth markers)
2. Otsu binary threshold → rock mask (rock white, gaps black)
3. Canny edge detection → crack/boundary signal
4. Column-wise projection: rock presence − edge penalty
5. Smooth + Otsu threshold on 1D score → piece boundaries

### Metrics
- **TCR** = (sum of all recovered lengths / 150 cm) × 100, capped at 100%
- **RQD** = (sum of lengths ≥ 10 cm / 150 cm) × 100, capped at 100%
- Noise filter: pieces < 5 cm removed
- Cap: pieces > 150 cm capped at 150 cm

---

## Project Structure

```
tcr-rqd-ai/
├── utils/
│   ├── run_segmentation.py   ← horizontal run detection
│   ├── piece_detection.py    ← core piece detection
│   ├── metrics.py            ← TCR/RQD + export (CSV/JSON/XLSX)
│   └── scale_detector.py     ← ruler/scale bar → px/cm
├── ui/
│   └── app.py                ← Streamlit web interface
├── pipeline.py               ← CLI orchestrator
├── notebooks/
│   ├── demo_pipeline.ipynb   ← Jupyter demo
│   └── 01_TCR_RQD_Training.py ← Colab training notebook
├── tests/
│   └── test_metrics.py       ← pytest unit tests
├── data/                     ← sample images (gitignored)
├── requirements.txt
└── README.md
```

---

## Testing

```bash
pytest tests/ -v
```

---

## Sample Output

```
==============================================================
  Image  : 1.png
  Scale  : 5.23 px/cm  [alternating_blocks]
==============================================================
   Run    TCR%    RQD%  Quality       Pieces
  ----  -------  -------  ------------  ------
     1     72.3     65.1  Fair               6
     2     85.4     78.2  Good               5
     3     68.9     55.7  Fair               7
     4     91.2     88.0  Good               4
     5     63.1     48.3  Poor               5
  ----  -------  -------  ------------  ------
   AVG     76.2     67.1  Fair              27
==============================================================
```

---

## Limitations & Known Issues

| Issue | Impact | Mitigation |
|-------|--------|-----------|
| Scale bar not detected | Falls back to w/120 estimate | Use high-res photos with clear ruler |
| Overlapping pieces | May merge into one detection | Tuned edge penalty helps; future YOLO model will improve |
| Highly fractured zones | May under-count small pieces | 5 cm noise filter removes debris |
| No depth marker OCR yet | Runs detected by projection, not by depth labels | Planned for future release |

---

## Roadmap

- [x] Run segmentation (projection + Otsu)
- [x] Piece detection (binary + Canny)
- [x] TCR/RQD calculation with quality classification
- [x] Scale bar detection
- [x] Streamlit UI with multi-image support
- [x] CLI pipeline
- [x] Multi-format export (JSON/CSV/XLSX)
- [x] Training notebook (YOLOv8-seg)
- [ ] Depth marker OCR (pytesseract)
- [ ] YOLO model integration for piece detection
- [ ] PDF report generation
- [ ] REST API (FastAPI)

---

## License

MIT — free to use, modify, and distribute with attribution.
