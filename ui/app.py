"""
TCR / RQD Automatic Calculator  —  Streamlit UI
=================================================
Launch:  streamlit run ui/app.py
"""

import io
import json
import os
import sys

import cv2
import numpy as np
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.run_segmentation import extract_runs
from utils.piece_detection import detect_pieces
from utils.metrics import compute_lengths, compute_metrics, rqd_quality, export_results
from utils.scale_detector import detect_scale

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


def _draw_boxes(img, boxes, lengths, color=(0, 255, 0), thickness=2):
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


# ---------- pipeline ----------

def process_image(image, run_length_cm=150.0, use_scale=True):
    scale_info = {"px_per_cm": 0.0, "method": "fixed", "confidence": 0.0}
    if use_scale:
        scale_info = detect_scale(image)

    runs = extract_runs(image)
    results = []

    for i, run in enumerate(runs):
        try:
            boxes = detect_pieces(run)
            lengths = compute_lengths(boxes, run, run_length_cm)
            metrics = compute_metrics(lengths, run_length_cm)
            annotated = _draw_boxes(run, boxes, lengths)
            results.append({
                "run_index": i,
                "run_image": run,
                "annotated_image": annotated,
                "boxes": boxes,
                "lengths": lengths,
                "metrics": metrics,
            })
        except Exception as e:
            results.append({"run_index": i, "error": str(e)})

    return results, scale_info


# ---------- page config ----------

st.set_page_config(page_title="TCR / RQD Calculator", page_icon="🪨", layout="wide")

# Page navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Select Mode", ["🤖 Automatic Mode", "✂️ Manual Runs Mode"])

if page == "✂️ Manual Runs Mode":
    # Import and run manual runs page
    import importlib.util
    spec = importlib.util.spec_from_file_location("manual_runs", os.path.join(os.path.dirname(__file__), "manual_runs.py"))
    manual_runs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manual_runs)
    manual_runs.main()
    st.stop()

# Automatic Mode (existing code)
st.title("🪨 TCR / RQD Automatic Calculator")
st.caption("Upload core-box images → detect runs & pieces → compute TCR & RQD.")

# ---------- sidebar ----------

with st.sidebar:
    st.header("⚙️ Settings")
    run_length_cm = st.number_input("Run length (cm)", 10.0, 500.0, 150.0, 10.0)
    rqd_min_cm = st.number_input("RQD min piece (cm)", 1.0, 50.0, 10.0, 1.0)
    use_scale = st.toggle("Auto scale detection", True)
    show_boxes = st.toggle("Show bounding boxes", True)
    show_lengths = st.toggle("Show piece lengths", True)
    st.divider()
    st.markdown("**Export formats**")
    exp_csv = st.checkbox("CSV", value=True)
    exp_json = st.checkbox("JSON", value=True)
    exp_xlsx = st.checkbox("XLSX", value=False)
    st.divider()
    st.markdown(
        "**Pipeline**\n"
        "1. Scale bar detection\n"
        "2. Run segmentation (Otsu + projection)\n"
        "3. Piece detection (binary + Canny)\n"
        "4. TCR / RQD calculation"
    )

# ---------- session state ----------

if "all_results" not in st.session_state:
    st.session_state.all_results = {}

# ---------- image source ----------

tab_upload, tab_sample = st.tabs(["📤 Upload Images", "📁 Sample Images"])

with tab_upload:
    uploaded_files = st.file_uploader(
        "Choose core-box images",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=True,
    )

with tab_sample:
    data_dir = os.path.join(ROOT, "data")
    samples = sorted(
        f for f in os.listdir(data_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ) if os.path.isdir(data_dir) else []
    if samples:
        chosen = st.selectbox("Select sample", samples)
        if st.button("Load sample", use_container_width=True):
            img = cv2.imread(os.path.join(data_dir, chosen))
            if img is not None:
                results, scale_info = process_image(img, run_length_cm, use_scale)
                st.session_state.all_results[chosen] = {
                    "image": img, "results": results, "scale": scale_info
                }
    else:
        st.info("No images in `data/`.")

# ---------- process uploaded ----------

if uploaded_files:
    proc_col, clear_col = st.columns([1, 5])
    with proc_col:
        run_btn = st.button("▶ Analyse", type="primary", use_container_width=True)
    with clear_col:
        if st.button("🗑 Clear results"):
            st.session_state.all_results = {}
            st.rerun()

    if run_btn:
        progress = st.progress(0, text="Analysing images…")
        for idx, uf in enumerate(uploaded_files):
            img_bytes = np.frombuffer(uf.read(), np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if img is None:
                st.warning(f"Cannot decode {uf.name}")
                continue
            results, scale_info = process_image(img, run_length_cm, use_scale)
            st.session_state.all_results[uf.name] = {
                "image": img, "results": results, "scale": scale_info
            }
            progress.progress((idx + 1) / len(uploaded_files),
                              text=f"Processed {idx + 1}/{len(uploaded_files)}")
        progress.empty()
        st.success(f"✅ Processed {len(st.session_state.all_results)} image(s)")

# ---------- display results ----------

if not st.session_state.all_results:
    st.info("👆 Upload or load images to get started.")
    st.stop()

# Global summary
st.divider()
st.subheader("📊 Summary — All Images")
summary_rows = []
for name, data in st.session_state.all_results.items():
    valid = [r for r in data["results"] if "error" not in r]
    if valid:
        tcrs = [r["metrics"]["tcr"] for r in valid]
        rqds = [r["metrics"]["rqd"] for r in valid]
        avg_rqd = float(np.mean(rqds))
        summary_rows.append({
            "Image": name,
            "Runs": len(valid),
            "Avg TCR %": round(float(np.mean(tcrs)), 1),
            "Avg RQD %": round(avg_rqd, 1),
            "Rock Quality": rqd_quality(avg_rqd),
            "Scale (px/cm)": round(data["scale"]["px_per_cm"], 2),
            "Scale Method": data["scale"]["method"],
        })
if summary_rows:
    st.dataframe(summary_rows, use_container_width=True)

# Per-image tabs
tabs = st.tabs(list(st.session_state.all_results.keys()))

for tab, (name, data) in zip(tabs, st.session_state.all_results.items()):
    with tab:
        results = data["results"]
        valid = [r for r in results if "error" not in r]
        if not valid:
            st.error("No runs detected.")
            continue

        tcr_vals = [r["metrics"]["tcr"] for r in valid]
        rqd_vals = [r["metrics"]["rqd"] for r in valid]
        total_pieces = sum(r["metrics"]["n_pieces"] for r in valid)
        avg_rqd = float(np.mean(rqd_vals))

        # Summary metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Avg TCR", f"{np.mean(tcr_vals):.1f}%")
        c2.metric("Avg RQD", f"{np.mean(rqd_vals):.1f}%")
        c3.metric("Quality", rqd_quality(avg_rqd))
        c4.metric("Runs", len(valid))
        c5.metric("Pieces", total_pieces)

        with st.expander("Original Image", expanded=False):
            st.image(_rgb(data["image"]), caption=name, use_container_width=True)

        st.divider()

        # Per-run results
        for r in results:
            if "error" in r:
                st.error(f"Run {r['run_index']+1}: {r['error']}")
                continue

            m = r["metrics"]
            qc = m.get("rqd_class", rqd_quality(m["rqd"]))
            qcolor = QUALITY_COLORS.get(qc, "#888")

            with st.expander(
                f"Run {r['run_index']+1} — TCR {m['tcr']}% | "
                f"RQD {m['rqd']}% ({qc}) | {m['n_pieces']} pieces",
                expanded=True,
            ):
                ic, dc = st.columns([3, 1])
                with ic:
                    disp = r["annotated_image"] if show_boxes else r["run_image"]
                    st.image(_rgb(disp), use_container_width=True)
                with dc:
                    st.metric("TCR", f"{m['tcr']:.1f}%")
                    st.metric("RQD", f"{m['rqd']:.1f}%")
                    st.markdown(
                        f"**Quality:** <span style='color:{qcolor};font-weight:bold'>"
                        f"{qc}</span>", unsafe_allow_html=True
                    )
                    st.metric("Pieces", m["n_pieces"])
                    st.metric("RQD-eligible", m["n_rqd_pieces"])
                    st.metric("Recovered", f"{m['total_length']:.1f} cm")

                    if show_lengths and r["lengths"]:
                        st.caption("Piece lengths:")
                        for j, l in enumerate(r["lengths"]):
                            tag = "✅" if l >= rqd_min_cm else "❌"
                            st.text(f"  #{j+1}: {l:.1f} cm {tag}")

        # Export section
        st.divider()
        st.subheader("📥 Export")

        export = {
            "source": name,
            "run_length_cm": run_length_cm,
            "n_runs": len(valid),
            "scale": {
                "px_per_cm": round(data["scale"]["px_per_cm"], 2),
                "method": data["scale"]["method"],
            },
            "summary": {
                "avg_tcr": round(float(np.mean(tcr_vals)), 2),
                "avg_rqd": round(float(np.mean(rqd_vals)), 2),
                "total_pieces": total_pieces,
                "rqd_class": rqd_quality(avg_rqd),
            },
            "runs": [
                {
                    "run": r["run_index"] + 1,
                    "lengths_cm": [round(l, 2) for l in r["lengths"]],
                    "metrics": r["metrics"],
                }
                for r in valid
            ],
        }

        exp_cols = st.columns(4)

        # JSON
        if exp_json:
            exp_cols[0].download_button(
                "⬇ JSON",
                json.dumps(export, indent=2),
                file_name=f"{name.rsplit('.', 1)[0]}_tcr_rqd.json",
                mime="application/json",
                use_container_width=True,
            )

        # CSV
        if exp_csv:
            rows = []
            for r in valid:
                m = r["metrics"]
                rows.append(
                    f"{r['run_index']+1},{m['tcr']},{m['rqd']},"
                    f"{m['total_length']},{m['rqd_length']},"
                    f"{m.get('rqd_class', '')},{m['n_pieces']},{m['n_rqd_pieces']}"
                )
            csv_str = "Run,TCR (%),RQD (%),TCR (cm),RQD (cm),Quality,Pieces,RQD Pieces\n"
            csv_str += "\n".join(rows)
            exp_cols[1].download_button(
                "⬇ CSV", csv_str,
                file_name=f"{name.rsplit('.', 1)[0]}_tcr_rqd.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Annotated image
        full_ann = data["image"].copy()
        # Stitch annotated runs back (simplified: just download first run annotated)
        if valid:
            _, ann_enc = cv2.imencode(".jpg", valid[0]["annotated_image"],
                                      [cv2.IMWRITE_JPEG_QUALITY, 90])
            exp_cols[2].download_button(
                "⬇ Annotated Image", ann_enc.tobytes(),
                file_name=f"{name.rsplit('.', 1)[0]}_annotated.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )

        # XLSX
        if exp_xlsx:
            try:
                import pandas as pd
                xlsx_rows = []
                for r in valid:
                    m = r["metrics"]
                    xlsx_rows.append({
                        "Run": r["run_index"] + 1,
                        "TCR (%)": m["tcr"],
                        "RQD (%)": m["rqd"],
                        "TCR (cm)": m["total_length"],
                        "RQD (cm)": m["rqd_length"],
                        "Rock Quality": m.get("rqd_class", ""),
                        "# Pieces": m["n_pieces"],
                        "# RQD Pieces": m["n_rqd_pieces"],
                    })
                xlsx_buf = io.BytesIO()
                pd.DataFrame(xlsx_rows).to_excel(xlsx_buf, index=False,
                                                  sheet_name="TCR_RQD",
                                                  engine="openpyxl")
                exp_cols[3].download_button(
                    "⬇ XLSX", xlsx_buf.getvalue(),
                    file_name=f"{name.rsplit('.', 1)[0]}_tcr_rqd.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except ImportError:
                exp_cols[3].info("Install openpyxl for XLSX export")
