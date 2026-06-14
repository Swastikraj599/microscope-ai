# app.py — MicroScope AI
# Part 1: imports, config, model loading (trimmed — no torch/SAM2)

import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image
import json, os, time, io
from huggingface_hub import hf_hub_download
import google.generativeai as genai
import plotly.graph_objects as go
import duckdb

st.set_page_config(
    page_title="MicroScope AI",
    page_icon="\U0001F52C",
    layout="wide"
)

HF_REPO = "swastikraj/microscope-ai"

# Risk level mapping per class (rough heuristic for water safety — illustrative only)
RISK_MAP = {
    'actinophrys': 'caution', 'arcella': 'safe', 'aspidisca': 'safe',
    'codosiga': 'safe', 'colpoda': 'caution', 'epistylis': 'safe',
    'euglypha': 'safe', 'paramecium': 'caution', 'rotifera': 'safe',
    'vorticella': 'safe', 'noctiluca': 'hazardous', 'ceratium': 'hazardous',
    'stentor': 'caution', 'siprostomum': 'caution', 'keratella_quadrala': 'safe',
    'euglena': 'caution', 'gymnodinium': 'hazardous', 'gonyaulax': 'hazardous',
    'phacus': 'caution', 'stylongchia': 'safe', 'synchaeta': 'safe',
}

# ---------- Model loading (cached) ----------
@st.cache_resource
def load_classifier():
    onnx_path = hf_hub_download(repo_id=HF_REPO, filename="microscope_ai.onnx")
    labels_path = hf_hub_download(repo_id=HF_REPO, filename="class_labels.json")
    with open(labels_path) as f:
        labels_data = json.load(f)
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    return sess, labels_data['classes']

# ---------- Preprocessing ----------
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(pil_img):
    img = pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[np.newaxis, :, :, :]
    return arr.astype(np.float32)

# ---------- Load everything ----------
classifier_sess, CLASSES = load_classifier()
SAM2_AVAILABLE = False  # disabled in this deploy — re-enable when GPU hosting available

# ---------- DuckDB session log (in-memory) ----------
if 'db' not in st.session_state:
    con = duckdb.connect(database=':memory:')
    con.execute("""
        CREATE TABLE detections (
            id INTEGER, timestamp VARCHAR, species VARCHAR,
            confidence DOUBLE, risk VARCHAR, segmentation_used BOOLEAN
        )
    """)
    st.session_state['db'] = con
    st.session_state['detection_count'] = 0

con = st.session_state['db']

# ---------- Sidebar status ----------
with st.sidebar:
    st.title("MicroScope AI")
    st.caption("Real-time environmental microorganism intelligence")
    st.divider()
    st.markdown("**System status**")
    st.markdown("- Classifier: MobileViT-v2 (active)")
    st.markdown("- Segmentation: disabled (no GPU)")
    st.caption("Running in classify-full-image mode")
    st.markdown("- Model accuracy: 93.65% (test set)")
    st.divider()

    # Try secrets first, fall back to manual input
    gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    if not gemini_key:
        gemini_key = st.text_input("Gemini API key", type="password", help="Used to generate diagnostic reports")
    if gemini_key:
        genai.configure(api_key=gemini_key)
        st.session_state['gemini_ready'] = True

st.markdown("# MicroScope AI")
st.caption("Upload a microscopy image to identify environmental microorganisms and assess water safety")

tab1, tab2, tab3 = st.tabs(["Analyze", "Dashboard", "Report"])

with tab1:
    st.subheader("Image input")
    col_input, col_result = st.columns([1, 1])

    with col_input:
        input_method = st.radio("Source", ["Upload image", "Camera"], horizontal=True)
        if input_method == "Upload image":
            uploaded = st.file_uploader("Microscopy image", type=['png','jpg','jpeg'])
        else:
            uploaded = st.camera_input("Capture from webcam")

        if uploaded:
            pil_img = Image.open(uploaded)
            st.image(pil_img, caption="Input image", use_container_width=True)

    with col_result:
        if uploaded:
            st.subheader("Detection result")
            with st.spinner("Analyzing..."):
                proc_img = pil_img

                # Classification
                input_tensor = preprocess(proc_img)
                logits = classifier_sess.run(None, {'input': input_tensor})[0][0]
                probs = np.exp(logits) / np.sum(np.exp(logits))
                top3_idx = np.argsort(probs)[::-1][:3]

                top_species = CLASSES[top3_idx[0]]
                top_conf = float(probs[top3_idx[0]])
                risk = RISK_MAP.get(top_species, 'caution')

            # Risk badge
            risk_colors = {'safe': 'green', 'caution': 'orange', 'hazardous': 'red'}
            st.markdown(f"### {top_species.replace('_',' ').title()}")
            st.markdown(f":{risk_colors[risk]}[**{risk.upper()}**] — confidence {top_conf:.1%}")

            st.markdown("**Top 3 predictions**")
            for idx in top3_idx:
                st.progress(float(probs[idx]), text=f"{CLASSES[idx].replace('_',' ').title()} — {probs[idx]:.1%}")

            # Log to DuckDB
            st.session_state['detection_count'] += 1
            con.execute(
                "INSERT INTO detections VALUES (?, ?, ?, ?, ?, ?)",
                [st.session_state['detection_count'],
                 time.strftime('%Y-%m-%d %H:%M:%S'),
                 top_species, top_conf, risk, SAM2_AVAILABLE]
            )

            # Store last result for Report tab
            st.session_state['last_result'] = {
                'species': top_species, 'confidence': top_conf,
                'risk': risk, 'top3': [(CLASSES[i], float(probs[i])) for i in top3_idx]
            }
        else:
            st.info("Upload an image or use the camera to begin analysis")

with tab2:
    st.subheader("Session analytics")

    row_count = con.execute("SELECT COUNT(*) FROM detections").fetchone()[0]

    if row_count == 0:
        st.info("No detections yet. Analyze an image in the Analyze tab to populate this dashboard.")
    else:
        # ---------- Summary metric cards ----------
        risk_counts = con.execute("""
            SELECT risk, COUNT(*) as cnt FROM detections GROUP BY risk
        """).fetchall()
        risk_dict = {r[0]: r[1] for r in risk_counts}

        avg_conf = con.execute("SELECT AVG(confidence) FROM detections").fetchone()[0]
        unique_species = con.execute("SELECT COUNT(DISTINCT species) FROM detections").fetchone()[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total scans", row_count)
        m2.metric("Unique species", unique_species)
        m3.metric("Avg confidence", f"{avg_conf:.1%}")
        m4.metric("Hazardous detections", risk_dict.get('hazardous', 0))

        st.divider()

        col_left, col_right = st.columns(2)

        # ---------- Species frequency bar chart ----------
        with col_left:
            st.markdown("**Species frequency**")
            species_freq = con.execute("""
                SELECT species, COUNT(*) as cnt
                FROM detections
                GROUP BY species ORDER BY cnt DESC
            """).fetchall()

            fig_bar = go.Figure(go.Bar(
                x=[r[1] for r in species_freq],
                y=[r[0].replace('_',' ').title() for r in species_freq],
                orientation='h',
                marker_color='#1D9E75'
            ))
            fig_bar.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(autorange='reversed'),
                xaxis_title="Detections",
                template='plotly_white'
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ---------- Risk distribution pie chart ----------
        with col_right:
            st.markdown("**Risk level distribution**")
            risk_order = ['safe', 'caution', 'hazardous']
            risk_colors_map = {'safe': '#639922', 'caution': '#BA7517', 'hazardous': '#A32D2D'}
            values = [risk_dict.get(r, 0) for r in risk_order]

            fig_pie = go.Figure(go.Pie(
                labels=[r.title() for r in risk_order],
                values=values,
                marker_colors=[risk_colors_map[r] for r in risk_order],
                hole=0.45
            ))
            fig_pie.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                template='plotly_white'
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()

        # ---------- Detection timeline ----------
        st.markdown("**Detection timeline**")
        timeline = con.execute("""
            SELECT timestamp, species, confidence, risk
            FROM detections ORDER BY id DESC LIMIT 20
        """).fetchall()

        fig_timeline = go.Figure(go.Scatter(
            x=[r[0] for r in timeline],
            y=[r[2] for r in timeline],
            mode='markers+lines',
            text=[r[1].replace('_',' ').title() for r in timeline],
            hovertemplate='%{text}<br>Confidence: %{y:.1%}<extra></extra>',
            marker=dict(
                size=10,
                color=[risk_colors_map[r[3]] for r in timeline]
            ),
            line=dict(color='#888780', width=1)
        ))
        fig_timeline.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="Confidence",
            yaxis_tickformat='.0%',
            template='plotly_white'
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        # ---------- Raw table ----------
        st.markdown("**Detection log**")
        log_rows = con.execute("""
            SELECT timestamp, species, confidence, risk, segmentation_used
            FROM detections ORDER BY id DESC
        """).fetchall()
        st.dataframe(
            {
                "Time": [r[0] for r in log_rows],
                "Species": [r[1].replace('_',' ').title() for r in log_rows],
                "Confidence": [f"{r[2]:.1%}" for r in log_rows],
                "Risk": [r[3].title() for r in log_rows],
                "Segmented": ["Yes" if r[4] else "No" for r in log_rows],
            },
            use_container_width=True,
            hide_index=True
        )

with tab3:
    st.subheader("Diagnostic report")

    if 'last_result' not in st.session_state:
        st.info("Analyze an image first — the report is generated from your most recent detection.")
    else:
        result = st.session_state['last_result']
        species = result['species'].replace('_',' ').title()
        confidence = result['confidence']
        risk = result['risk']

        st.markdown(f"**Latest detection:** {species} ({confidence:.1%} confidence, risk: {risk.title()})")

        if not st.session_state.get('gemini_ready'):
            st.warning("Enter a Gemini API key in the sidebar to generate the narrative report.")
        else:
            if st.button("Generate report"):
                with st.spinner("Generating diagnostic report..."):
                    try:
                        gem_model = genai.GenerativeModel('gemini-1.5-flash')
                        prompt = f"""
You are assisting a water quality technician. A microscopy classifier identified
the organism "{species}" with {confidence:.1%} confidence. Its assigned
preliminary risk category is "{risk}".

Write a short structured report with these sections:
1. Species profile (2-3 sentences on what this organism is)
2. Water quality implication (what its presence may indicate)
3. Recommended action (practical next step for a field technician)

Keep it factual, concise, and clearly note this is a preliminary
AI-assisted screening, not a substitute for lab confirmation.
"""
                        response = gem_model.generate_content(prompt)
                        st.session_state['report_text'] = response.text
                    except Exception as e:
                        st.error(f"Gemini request failed: {e}")

            if 'report_text' in st.session_state:
                st.markdown("---")
                st.markdown(st.session_state['report_text'])

        st.divider()

        # ---------- PDF export ----------
        st.markdown("**Export session report**")
        if st.button("Generate PDF"):
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "MicroScope AI - Session Report", ln=True)

            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 8, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.ln(4)

            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Latest detection", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 7, f"Species: {species}", ln=True)
            pdf.cell(0, 7, f"Confidence: {confidence:.1%}", ln=True)
            pdf.cell(0, 7, f"Risk level: {risk.title()}", ln=True)
            pdf.ln(4)

            if 'report_text' in st.session_state:
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, "Diagnostic notes", ln=True)
                pdf.set_font("Helvetica", "", 10)
                # latin-1 strip — fpdf core fonts don't support full unicode
                clean_text = st.session_state['report_text'].encode('latin-1', 'ignore').decode('latin-1')
                pdf.multi_cell(0, 6, clean_text)
                pdf.ln(4)

            # Detection log table
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Session detection log", ln=True)
            pdf.set_font("Helvetica", "", 9)
            log_rows = con.execute("""
                SELECT timestamp, species, confidence, risk
                FROM detections ORDER BY id
            """).fetchall()
            for r in log_rows:
                line = f"{r[0]} | {r[1].replace('_',' ').title()} | {r[2]:.1%} | {r[3].title()}"
                pdf.cell(0, 6, line, ln=True)

            pdf_bytes = bytes(pdf.output(dest='S'))
            st.download_button(
                "Download PDF report",
                data=pdf_bytes,
                file_name=f"microscope_ai_report_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
