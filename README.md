# TCR / RQD Automatic Calculator

A production-ready computer vision pipeline for automatically calculating Total Core Recovery (TCR) and Rock Quality Designation (RQD) from borehole core box images.

## Features

- **Run Segmentation**: Detects horizontal core runs using projection-based Otsu thresholding
- **Piece Detection**: Identifies individual core pieces using combined binary mask + Canny edge detection
- **Metric Calculation**: Computes TCR and RQD with configurable parameters
- **Web UI**: Streamlit-based interface for upload, processing, and results visualization
- **JSON Export**: Export results with per-run metrics and piece lengths

## Quick Start

### Installation

```bash
pip install opencv-python numpy matplotlib streamlit
```

### Web UI

```bash
streamlit run ui/app.py
```

### Python API

```python
from utils.run_segmentation import extract_runs
from utils.piece_detection import detect_pieces
from utils.metrics import compute_lengths, compute_metrics

# Load image
image = cv2.imread('path/to/core_box.png')

# Process pipeline
runs = extract_runs(image)
results = []
for run in runs:
    boxes = detect_pieces(run)
    lengths = compute_lengths(boxes, run, run_length_cm=150.0)
    metrics = compute_metrics(lengths, run_length_cm=150.0)
    results.append({'boxes': boxes, 'lengths': lengths, 'metrics': metrics})
```

## Architecture

- `utils/run_segmentation.py` - Horizontal run detection using row-wise projection
- `utils/piece_detection.py` - Core piece detection using column-wise projection + Canny edges
- `utils/metrics.py` - TCR/RQD calculation from piece lengths
- `ui/app.py` - Streamlit web interface
- `notebooks/demo_pipeline.ipynb` - Jupyter demo notebook

## Algorithm Details

### Run Segmentation
1. Convert to grayscale + Gaussian blur
2. Compute horizontal projection (row-wise mean)
3. Otsu threshold on 1D projection signal
4. Split at dark valleys (metal separators)

### Piece Detection
1. Crop margins (labels, tray edges, depth markers)
2. Otsu binary threshold → rock mask
3. Canny edge detection → crack signal
4. Column-wise projection: rock presence - edge penalty
5. Otsu threshold on 1D score → piece boundaries

### Metrics
- **TCR (%)** = (sum of all recovered lengths / 150 cm) × 100
- **RQD (%)** = (sum of lengths ≥ 10 cm / 150 cm) × 100

## Parameters

- Run length: 150 cm (configurable)
- Minimum piece length for RQD: 10 cm
- Noise filter: < 5 cm pieces removed
- Maximum piece length: capped at 150 cm

## Sample Results

```
Summary — 5 runs detected
Avg TCR: 71.2%
Avg RQD: 67.4%
Total Runs: 5
Total Pieces: 27
```

## Project Structure

```
tcr-rqd-ai/
├── utils/
│   ├── run_segmentation.py
│   ├── piece_detection.py
│   └── metrics.py
├── ui/
│   └── app.py
├── notebooks/
│   └── demo_pipeline.ipynb
├── data/
│   └── sample_images/
├── outputs/
└── tests/
```

## License

MIT License
