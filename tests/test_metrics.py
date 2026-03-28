"""
tests/test_metrics.py
─────────────────────
Unit tests for metrics module.

Run with:  pytest tests/test_metrics.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from utils.metrics import compute_lengths, compute_metrics, rqd_quality, export_results


# ──────────────────────────────────────────────────────────────────────────────
# rqd_quality tests
# ──────────────────────────────────────────────────────────────────────────────

class TestRQDQuality:
    def test_excellent(self):     assert rqd_quality(95) == "Excellent"
    def test_good(self):          assert rqd_quality(80) == "Good"
    def test_fair(self):          assert rqd_quality(60) == "Fair"
    def test_poor(self):          assert rqd_quality(35) == "Poor"
    def test_very_poor(self):     assert rqd_quality(10) == "Very Poor"
    def test_boundary_90(self):   assert rqd_quality(90) == "Excellent"
    def test_boundary_75(self):   assert rqd_quality(75) == "Good"
    def test_boundary_50(self):   assert rqd_quality(50) == "Fair"
    def test_boundary_25(self):   assert rqd_quality(25) == "Poor"
    def test_zero(self):          assert rqd_quality(0) == "Very Poor"
    def test_hundred(self):       assert rqd_quality(100) == "Excellent"


# ──────────────────────────────────────────────────────────────────────────────
# compute_lengths tests
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeLengths:
    def _make_run(self, width_px=1000, height=100):
        return np.zeros((height, width_px, 3), dtype=np.uint8)

    def test_basic(self):
        run = self._make_run(1500)
        # box width 150px → 150 * (150/1500) = 15 cm
        boxes = [(0, 0, 150, 50)]
        lengths = compute_lengths(boxes, run, 150.0)
        assert len(lengths) == 1
        assert abs(lengths[0] - 15.0) < 0.1

    def test_filters_small(self):
        run = self._make_run(1500)
        # box width 30px → 30 * (150/1500) = 3 cm → filtered out (< 5 cm)
        boxes = [(0, 0, 30, 50)]
        lengths = compute_lengths(boxes, run, 150.0)
        assert len(lengths) == 0

    def test_caps_large(self):
        run = self._make_run(1500)
        # box width 2000px → 200 cm → capped at 150
        boxes = [(0, 0, 2000, 50)]
        lengths = compute_lengths(boxes, run, 150.0)
        assert len(lengths) == 1
        assert lengths[0] == 150.0

    def test_empty_boxes(self):
        run = self._make_run()
        assert compute_lengths([], run) == []

    def test_none_image(self):
        assert compute_lengths([(0, 0, 100, 50)], None) == []

    def test_multiple_boxes(self):
        run = self._make_run(1500)
        boxes = [(0, 0, 300, 50), (400, 0, 200, 50)]
        lengths = compute_lengths(boxes, run, 150.0)
        assert len(lengths) == 2
        assert abs(lengths[0] - 30.0) < 0.1
        assert abs(lengths[1] - 20.0) < 0.1


# ──────────────────────────────────────────────────────────────────────────────
# compute_metrics tests
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_basic(self):
        m = compute_metrics([20, 15, 10], run_length_cm=150.0)
        assert abs(m["tcr"] - 30.0) < 0.1
        assert abs(m["rqd"] - 30.0) < 0.1
        assert m["n_pieces"] == 3
        assert m["n_rqd_pieces"] == 3

    def test_mixed_rqd(self):
        m = compute_metrics([20, 8, 5], run_length_cm=150.0)
        assert m["n_pieces"] == 3
        assert m["n_rqd_pieces"] == 1  # only 20 cm qualifies
        assert abs(m["rqd"] - (20 / 150 * 100)) < 0.1

    def test_all_small(self):
        m = compute_metrics([5, 7, 3, 9])
        assert m["rqd"] == 0.0
        assert m["n_rqd_pieces"] == 0

    def test_empty(self):
        m = compute_metrics([])
        assert m["tcr"] == 0.0
        assert m["rqd"] == 0.0
        assert m["n_pieces"] == 0

    def test_capped_at_100(self):
        m = compute_metrics([200])
        assert m["tcr"] <= 100.0

    def test_has_rqd_class(self):
        m = compute_metrics([100, 40])
        assert "rqd_class" in m
        assert m["rqd_class"] in ["Excellent", "Good", "Fair", "Poor", "Very Poor"]

    def test_rqd_class_excellent(self):
        m = compute_metrics([140], run_length_cm=150.0)
        assert m["rqd_class"] == "Excellent"

    def test_rqd_class_poor(self):
        m = compute_metrics([40], run_length_cm=150.0)
        # 40/150 * 100 = 26.67% → Poor
        assert m["rqd_class"] == "Poor"


# ──────────────────────────────────────────────────────────────────────────────
# export_results tests
# ──────────────────────────────────────────────────────────────────────────────

class TestExport:
    def _sample_results(self):
        return {
            "source": "test.png",
            "n_runs": 1,
            "runs": [
                {
                    "run": 1,
                    "metrics": {
                        "tcr": 50.0,
                        "rqd": 40.0,
                        "total_length": 75.0,
                        "rqd_length": 60.0,
                        "rqd_class": "Poor",
                        "n_pieces": 3,
                        "n_rqd_pieces": 2,
                    },
                }
            ],
        }

    def test_json_export(self, tmp_path):
        import json
        p = export_results(self._sample_results(), tmp_path / "out", "json")
        assert p.exists()
        data = json.loads(p.read_text())
        assert "runs" in data

    def test_csv_export(self, tmp_path):
        p = export_results(self._sample_results(), tmp_path / "out", "csv")
        assert p.exists()
        text = p.read_text()
        assert "TCR" in text
        assert "Run" in text
