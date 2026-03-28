"""
Manual Runs Mode — Streamlit Page for Cropped Run Images
========================================================
Processes manually cropped run images from data/cropped_images/
"""

import io
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.piece_detection import detect_pieces
from utils.metrics import compute_lengths, compute_metrics, rqd_quality, export_results

# ---------- helpers ----------

QUALITY_COLORS = {
    "Excellent": "#2dc653",
    "Good": "#8ecf6e",
    "Fair": "#f9c74f",
    "Poor": "#f77f00",
    "Very Poor": "#d62828",
}


def _rgb(img):
    if img is None:
        return img
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _draw_boxes(img, boxes, lengths, thickness=2):
    canvas = img.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        # RQD-eligible pieces in green, others in orange
        is_rqd = i < len(lengths) and lengths[i] >= 10
        c = (0, 200, 0) if is_rqd else (0, 100, 255)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), c, thickness)
        if i < len(lengths):
            label = f"{lengths[i]:.1f}cm"
            label_y = max(y - 8, 18)
            sc = max(0.35, min(0.55, h / 100))
            cv2.putText(canvas, label, (x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, sc, c, 1, cv2.LINE_AA)
    return canvas


def _parse_filename(filename):
    """Parse '1.1.png' → (1, 1)"""
    stem = Path(filename).stem
    try:
        img_id, run_id = stem.split('.')
        return int(img_id), int(run_id)
    except (ValueError, AttributeError):
        return None, None


def _group_images_by_original(cropped_dir):
    """Group images by original image ID and sort runs correctly."""
    groups = {}
    
    if not cropped_dir.exists():
        return groups
    
    for file_path in sorted(cropped_dir.glob("*.png")):
        img_id, run_id = _parse_filename(file_path.name)
        if img_id is not None and run_id is not None:
            if img_id not in groups:
                groups[img_id] = []
            groups[img_id].append((run_id, file_path))
    
    # Sort runs within each group
    for img_id in groups:
        groups[img_id].sort(key=lambda x: x[0])  # Sort by run_id
    
    return groups


@st.cache_data(show_spinner=False)
def _process_run_cached(image_path: Path, run_length_cm: float = 150.0) -> Tuple[Optional[Dict], Optional[str]]:
    """Process a single cropped run image (cached)."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return None, "Failed to load image"
        
        # Get image stats
        h, w = image.shape[:2]
        
        boxes = detect_pieces(image)
        lengths = compute_lengths(boxes, image, run_length_cm)
        metrics = compute_metrics(lengths, run_length_cm)
        annotated = _draw_boxes(image, boxes, lengths)
        
        return {
            "image_path": image_path,
            "image": image,
            "annotated": annotated,
            "boxes": boxes,
            "lengths": lengths,
            "metrics": metrics,
            "image_stats": {
                "width": w,
                "height": h,
                "aspect_ratio": w/h,
                "px_per_cm": run_length_cm / w * 100,  # approx
            }
        }, None
    except Exception as e:
        return None, str(e)


def _process_run(image_path, run_length_cm=150.0):
    """Process a single cropped run image (non-cached version)."""
    return _process_run_cached(image_path, run_length_cm)


def _calculate_group_summary(run_results):
    """Calculate summary for a group of runs."""
    if not run_results:
        return {"avg_tcr": 0, "avg_rqd": 0, "total_pieces": 0, "rqd_class": "Very Poor"}
    
    tcr_vals = [r["metrics"]["tcr"] for r in run_results]
    rqd_vals = [r["metrics"]["rqd"] for r in run_results]
    total_pieces = sum(r["metrics"]["n_pieces"] for r in run_results)
    
    avg_tcr = float(np.mean(tcr_vals)) if tcr_vals else 0.0
    avg_rqd = float(np.mean(rqd_vals)) if rqd_vals else 0.0
    
    return {
        "avg_tcr": round(avg_tcr, 2),
        "avg_rqd": round(avg_rqd, 2),
        "total_pieces": total_pieces,
        "rqd_class": rqd_quality(avg_rqd),
        "n_runs": len(run_results),
    }


def _export_all_results(all_results: Dict, run_length_cm: float) -> Dict:
    """Prepare export data for all processed images."""
    export_data = {
        "mode": "manual_runs",
        "run_length_cm": run_length_cm,
        "total_images": len(all_results),
        "summary": {},
        "images": {},
    }
    
    # Global summary
    all_runs = []
    for img_id, runs in all_results.items():
        all_runs.extend(runs)
    
    if all_runs:
        tcr_vals = [r["metrics"]["tcr"] for r in all_runs]
        rqd_vals = [r["metrics"]["rqd"] for r in all_runs]
        total_pieces = sum(r["metrics"]["n_pieces"] for r in all_runs)
        
        export_data["summary"] = {
            "avg_tcr": round(float(np.mean(tcr_vals)), 2),
            "avg_rqd": round(float(np.mean(rqd_vals)), 2),
            "total_pieces": total_pieces,
            "total_runs": len(all_runs),
            "rqd_class": rqd_quality(float(np.mean(rqd_vals))),
        }
    
    # Per-image data
    for img_id, runs in all_results.items():
        summary = _calculate_group_summary(runs)
        export_data["images"][str(img_id)] = {
            "summary": summary,
            "runs": [
                {
                    "run": i + 1,
                    "lengths_cm": [round(l, 2) for l in r["lengths"]],
                    "metrics": r["metrics"],
                    "image_stats": r.get("image_stats", {}),
                }
                for i, r in enumerate(runs)
            ],
        }
    
    return export_data


def _create_download_buttons(export_data: Dict):
    """Create download buttons for different formats."""
    col1, col2, col3 = st.columns(3)
    
    # JSON export
    with col1:
        json_data = json.dumps(export_data, indent=2)
        st.download_button(
            "📥 Download JSON",
            json_data,
            file_name="manual_runs_results.json",
            mime="application/json",
            use_container_width=True,
        )
    
    # CSV export
    with col2:
        if export_data["images"]:
            csv_lines = ["Image,Run,TCR(%),RQD(%),TCR(cm),RQD(cm),Quality,Pieces,RQD_Pieces"]
            for img_id, img_data in export_data["images"].items():
                for run_data in img_data["runs"]:
                    m = run_data["metrics"]
                    csv_lines.append(
                        f"{img_id},{run_data['run']},{m['tcr']},{m['rqd']},"
                        f"{m['total_length']},{m['rqd_length']},{m.get('rqd_class', '')},"
                        f"{m['n_pieces']},{m['n_rqd_pieces']}"
                    )
            csv_data = "\n".join(csv_lines)
            st.download_button(
                "📥 Download CSV",
                csv_data,
                file_name="manual_runs_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
    
    # XLSX export (if pandas available)
    with col3:
        try:
            import pandas as pd
            # Flatten data for DataFrame
            rows = []
            for img_id, img_data in export_data["images"].items():
                for run_data in img_data["runs"]:
                    m = run_data["metrics"]
                    rows.append({
                        "Image": img_id,
                        "Run": run_data["run"],
                        "TCR (%)": m["tcr"],
                        "RQD (%)": m["rqd"],
                        "TCR (cm)": m["total_length"],
                        "RQD (cm)": m["rqd_length"],
                        "Rock Quality": m.get("rqd_class", ""),
                        "# Pieces": m["n_pieces"],
                        "# RQD Pieces": m["n_rqd_pieces"],
                    })
            df = pd.DataFrame(rows)
            xlsx_buf = io.BytesIO()
            with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Manual_Runs", index=False)
                # Add summary sheet
                summary_df = pd.DataFrame([export_data["summary"]])
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
            
            st.download_button(
                "📥 Download XLSX",
                xlsx_buf.getvalue(),
                file_name="manual_runs_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except ImportError:
            st.info("Install pandas/openpyxl for XLSX export")


# ---------- main page ----------

def main():
    st.set_page_config(page_title="Manual Runs Mode", page_icon="✂️", layout="wide")
    st.title("✂️ Manual Runs Mode")
    st.caption("Process manually cropped run images from data/cropped_images/")
    
    # Settings
    with st.sidebar:
        st.header("⚙️ Settings")
        run_length_cm = st.number_input("Run length (cm)", 10.0, 500.0, 150.0, 10.0)
        rqd_min_cm = st.number_input("RQD min piece (cm)", 1.0, 50.0, 10.0, 1.0)
        show_boxes = st.toggle("Show bounding boxes", True)
        show_lengths = st.toggle("Show piece lengths", True)
        show_stats = st.toggle("Show image stats", False)
        use_cache = st.toggle("Use caching (faster)", True, help="Cache processed images for faster reload")
        st.divider()
        st.markdown("**Pipeline**\n1. Load cropped runs\n2. Detect pieces\n3. Compute TCR/RQD")
    
    # Cache control
    if not use_cache:
        _process_run_cached.clear()
        st.info("Cache cleared - processing will be slower")
    
    # Load and group images
    cropped_dir = Path(ROOT) / "data" / "cropped_images"
    image_groups = _group_images_by_original(cropped_dir)
    
    if not image_groups:
        st.warning("No cropped images found in `data/cropped_images/`")
        st.info("Expected format: `1.1.png`, `1.2.png`, etc. (image_id.run_id.png)")
        st.stop()
    
    # Show directory info
    with st.expander("📁 Directory Info", expanded=False):
        total_files = sum(len(runs) for runs in image_groups.values())
        st.write(f"**Directory**: `{cropped_dir}`")
        st.write(f"**Total images**: {len(image_groups)}")
        st.write(f"**Total runs**: {total_files}")
        st.write("**Image groups**:")
        for img_id in sorted(image_groups.keys()):
            st.write(f"  • Image {img_id}: {len(image_groups[img_id])} runs")
    
    # Process all images with progress bar
    st.divider()
    st.subheader("� Processing Images")
    
    if st.button("🚀 Process All Images", type="primary"):
        progress_bar = st.progress(0, text="Starting...")
        status_text = st.empty()
        
        all_results = {}
        total_runs = sum(len(runs) for runs in image_groups.values())
        processed = 0
        
        for img_id in sorted(image_groups.keys()):
            runs = image_groups[img_id]
            run_results = []
            
            for run_id, file_path in runs:
                status_text.text(f"Processing Image {img_id}, Run {run_id}...")
                result, error = _process_run(file_path, run_length_cm)
                if result:
                    run_results.append(result)
                processed += 1
                progress_bar.progress(processed / total_runs, 
                                   text=f"Processed {processed}/{total_runs} runs")
            
            if run_results:
                all_results[img_id] = run_results
        
        progress_bar.empty()
        status_text.empty()
        
        # Store results in session state
        st.session_state.manual_results = all_results
        st.session_state.run_length_cm = run_length_cm
        st.success(f"✅ Processed {len(all_results)} images with {total_runs} runs!")
    
    # Display results if available
    if "manual_results" in st.session_state:
        all_results = st.session_state.manual_results
        run_length_cm = st.session_state.run_length_cm
        
        # Summary across all images
        st.divider()
        st.subheader("📊 Summary — All Images")
        
        all_runs = []
        for img_id, runs in all_results.items():
            all_runs.extend(runs)
        
        if all_runs:
            tcr_vals = [r["metrics"]["tcr"] for r in all_runs]
            rqd_vals = [r["metrics"]["rqd"] for r in all_runs]
            total_pieces = sum(r["metrics"]["n_pieces"] for r in all_runs)
            avg_rqd = float(np.mean(rqd_vals))
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Images", len(all_results))
            c2.metric("Total Runs", len(all_runs))
            c3.metric("Avg TCR", f"{np.mean(tcr_vals):.1f}%")
            c4.metric("Avg RQD", f"{np.mean(rqd_vals):.1f}%")
            c5.metric("Total Pieces", total_pieces)
            
            # Quality distribution
            quality_counts = {}
            for r in all_runs:
                qc = r["metrics"].get("rqd_class", "Unknown")
                quality_counts[qc] = quality_counts.get(qc, 0) + 1
            
            st.write("**Rock Quality Distribution:**")
            qc_cols = st.columns(len(quality_counts))
            for i, (quality, count) in enumerate(quality_counts.items()):
                with qc_cols[i]:
                    color = QUALITY_COLORS.get(quality, "#888")
                    st.markdown(
                        f"<div style='background-color:{color};color:white;"
                        f"padding:8px;border-radius:4px;text-align:center'>"
                        f"<strong>{quality}</strong><br>{count} runs</div>",
                        unsafe_allow_html=True
                    )
        
        # Export buttons
        st.divider()
        st.subheader("💾 Export Results")
        export_data = _export_all_results(all_results, run_length_cm)
        _create_download_buttons(export_data)
        
        # Process each image group
        st.divider()
        st.subheader("🖼️ Individual Images")
        
        for img_id in sorted(all_results.keys()):
            run_results = all_results[img_id]
            summary = _calculate_group_summary(run_results)
            qc = summary["rqd_class"]
            qcolor = QUALITY_COLORS.get(qc, "#888")
            
            # Expander for this image
            with st.expander(
                f"📁 Image {img_id} — {summary['n_runs']} runs — "
                f"TCR {summary['avg_tcr']}% | RQD {summary['avg_rqd']}% ({qc}) | "
                f"{summary['total_pieces']} pieces",
                expanded=False,
            ):
                # Summary metrics and stats
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Avg TCR", f"{summary['avg_tcr']:.1f}%")
                sc2.metric("Avg RQD", f"{summary['avg_rqd']:.1f}%")
                sc3.metric("Quality", qc)
                sc4.metric("Total Pieces", summary["total_pieces"])
                
                # Image stats if enabled
                if show_stats and run_results:
                    with st.expander("📊 Image Statistics", expanded=False):
                        stats_data = []
                        for i, r in enumerate(run_results):
                            stats = r.get("image_stats", {})
                            stats_data.append({
                                "Run": i + 1,
                                "Width": stats.get("width", "N/A"),
                                "Height": stats.get("height", "N/A"),
                                "Aspect Ratio": f"{stats.get('aspect_ratio', 0):.2f}" if stats.get("aspect_ratio") else "N/A",
                                "px/cm": f"{stats.get('px_per_cm', 0):.1f}" if stats.get("px_per_cm") else "N/A",
                            })
                        st.dataframe(stats_data, use_container_width=True)
                
                # Display each run
                for i, run_result in enumerate(run_results):
                    m = run_result["metrics"]
                    
                    with st.expander(
                        f"🔹 Run {i+1} — TCR {m['tcr']}% | RQD {m['rqd']}% | {m['n_pieces']} pieces",
                        expanded=False,
                    ):
                        rc1, rc2 = st.columns([3, 1])
                        
                        with rc1:
                            disp = run_result["annotated"] if show_boxes else run_result["image"]
                            st.image(_rgb(disp), use_container_width=True)
                        
                        with rc2:
                            st.metric("TCR", f"{m['tcr']:.1f}%")
                            st.metric("RQD", f"{m['rqd']:.1f}%")
                            st.metric("Pieces", m["n_pieces"])
                            st.metric("RQD-eligible", m["n_rqd_pieces"])
                            st.metric("Recovered", f"{m['total_length']:.1f} cm")
                            
                            if show_lengths and run_result["lengths"]:
                                st.caption("Piece lengths:")
                                for j, l in enumerate(run_result["lengths"]):
                                    tag = "✅" if l >= rqd_min_cm else "❌"
                                    st.text(f"  #{j+1}: {l:.1f} cm {tag}")
    else:
        st.info("👆 Click 'Process All Images' to start analysis")


if __name__ == "__main__":
    main()
