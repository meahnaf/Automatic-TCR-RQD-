"""
TCR / RQD Automatic Calculator  —  Streamlit UI
=================================================
Launch:  streamlit run ui/app.py
"""

import io, json, os, sys
import cv2
import numpy as np
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.run_segmentation import extract_runs
from utils.piece_detection import detect_pieces
from utils.metrics import compute_lengths, compute_metrics

# ---------- helpers ----------

def _rgb(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _draw_boxes(img, boxes, lengths, color=(0, 255, 0), thickness=2):
    canvas = img.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (0,255,0), 2)
        if i < len(lengths):
            label = f"{lengths[i]:.1f} cm"
            # Position label above piece, not at fixed y=20
            label_y = max(y - 8, 20)
            cv2.putText(canvas, label, (x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    return canvas


# ---------- pipeline ----------

def process_image(image, run_length_cm=150):
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
            results.append({
                "run_index": i,
                "error": str(e)
            })
    return results

# ---------- page config ----------

st.set_page_config(page_title="TCR / RQD Calculator", page_icon="🪨", layout="wide")
st.title("🪨 TCR / RQD Automatic Calculator")
st.caption("Upload a core box image → detect runs & pieces → compute TCR & RQD.")

# ---------- sidebar ----------

with st.sidebar:
    st.header("Settings")
    run_length_cm = st.number_input("Run length (cm)", 10.0, 500.0, 150.0, 10.0)
    show_boxes = st.toggle("Show bounding boxes", True)
    show_lengths = st.toggle("Show piece lengths", True)
    st.divider()
    st.markdown("**Pipeline**\n1. Run segmentation\n2. Piece detection\n3. TCR / RQD")

# ---------- image source ----------

tab_upload, tab_sample = st.tabs(["📤 Upload", "📁 Samples"])
image = None
source_name = ""

with tab_upload:
    uploaded = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
    source_name = ""

    if uploaded:
        source_name = uploaded.name
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

with tab_sample:
    data_dir = os.path.join(ROOT, "data")
    samples = sorted(f for f in os.listdir(data_dir)
                     if f.lower().endswith((".png",".jpg",".jpeg"))) if os.path.isdir(data_dir) else []
    if samples:
        chosen = st.selectbox("Select sample", samples)
        if st.button("Load sample", use_container_width=True):
            image = cv2.imread(os.path.join(data_dir, chosen))
            source_name = chosen
    else:
        st.info("No images in `data/`.")

# ---------- processing ----------

if image is not None:
    st.divider()
    with st.expander("Original Image", expanded=False):
        st.image(_rgb(image), caption=source_name, use_container_width=True)

    with st.spinner("Processing..."):
        results = process_image(image, run_length_cm)

    if not results:
        st.error("No runs detected.")
        st.stop()

    n_runs = len(results)
    tcr_vals = [r["metrics"]["tcr"] for r in results]
    rqd_vals = [r["metrics"]["rqd"] for r in results]
    total_pieces = sum(r["metrics"]["n_pieces"] for r in results)

    # summary
    st.subheader(f"Summary — {n_runs} runs detected")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg TCR", f"{np.mean(tcr_vals):.1f}%")
    c2.metric("Avg RQD", f"{np.mean(rqd_vals):.1f}%")
    c3.metric("Total Runs", n_runs)
    c4.metric("Total Pieces", total_pieces)
    st.divider()

    for r in results:
        if "error" in r:
            st.error(f"Run {r['run_index']+1} failed: {r['error']}")
            continue
            
        m = r["metrics"]

        with st.expander(
            f"Run {r['run_index']+1} — TCR {m['tcr']}% | RQD {m['rqd']}% | {m['n_pieces']} pieces",
            expanded=True,
        ):
            col1, col2 = st.columns([3,1])
            
            with col1:
                st.image(_rgb(r["annotated_image"]), use_container_width=True)
            
            with col2:
                st.metric("TCR", f"{m['tcr']:.2f}%")
                st.metric("RQD", f"{m['rqd']:.2f}%")
                st.metric("Pieces", m["n_pieces"])
                st.metric("RQD-eligible", m["n_rqd_pieces"])
                st.metric("Recovered", f"{m['total_length']:.1f} cm")
                
                st.write("Lengths:")
                for l in r["lengths"]:
                    tag = "✅" if l >= 10 else "❌"
                    st.write(f"{l:.1f} cm {tag}")

    st.divider()
    
    export = {
        "source": source_name,
        "run_length_cm": run_length_cm,
        "n_runs": len(results),
        "summary": {
            "avg_tcr": round(float(np.mean(tcr_vals)), 2),
            "avg_rqd": round(float(np.mean(rqd_vals)), 2),
            "total_pieces": total_pieces,
        },
        "runs": [
            {
                "run": r["run_index"] + 1,
                "lengths_cm": [round(l, 2) for l in r["lengths"]],
                "metrics": r["metrics"],
            }
            for r in results
        ]
    }

    st.download_button(
        "📥 Download results (JSON)",
        json.dumps(export, indent=2),
        file_name="tcr_rqd_results.json",
        mime="application/json"
    )
