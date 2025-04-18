import warnings
warnings.filterwarnings("ignore")

import io
import tempfile
import base64

import streamlit as st
import whisper
import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

# --- Helpers & Caching ---------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_whisper(model_name="base"):
    return whisper.load_model(model_name)

@st.cache_data(show_spinner=False)
def extract_features(audio_bytes: bytes, language: str, model_name: str):
    # write temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        seg.export(tmp.name, format="wav")
        path = tmp.name

    # Whisper ASR
    model = load_whisper(model_name)
    res = model.transcribe(path, language=language)
    transcript = res["text"].strip()

    # Pause detection
    y, sr = librosa.load(path, sr=None, mono=True)
    intervals = librosa.effects.split(y, top_db=25)
    speech_dur   = sum((e - s) for s, e in intervals) / sr
    total_dur    = len(y) / sr
    total_pause  = total_dur - speech_dur
    num_pauses   = max(len(intervals) - 1, 0)

    # MFCC (optional)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfcc.mean(axis=1)
    mfcc_stds  = mfcc.std(axis=1)

    feats = {
        "filename":      tmp.name.split("/")[-1],
        "transcript":    transcript,
        "total_duration": total_dur,
        "total_pause":   total_pause,
        "num_pauses":    num_pauses,
    }
    for i in range(13):
        feats[f"mfcc_{i+1}_mean"] = float(mfcc_means[i])
        feats[f"mfcc_{i+1}_std"]  = float(mfcc_stds[i])
    return feats

@st.cache_data(show_spinner=False)
def score_df(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["total_pause","num_pauses"]]
    db  = DBSCAN(eps=0.5, min_samples=2).fit(X)
    iso = IsolationForest(contamination=0.1).fit(X)
    df["dbscan_label"] = db.labels_
    df["iso_score"]    = iso.decision_function(X)

    pmax, nmax, imax = df.total_pause.max(), df.num_pauses.max(), df.iso_score.max()
    df["pause_norm"] = df.total_pause.apply(lambda v: v/pmax if pmax>0 else 0)
    df["count_norm"] = df.num_pauses.apply(lambda v: v/nmax if nmax>0 else 0)
    df["iso_norm"]   = df.iso_score.apply(lambda v: 1-(v/imax) if imax>0 else 0).fillna(0)

    w1, w2, w3 = 0.5, 0.3, 0.2
    df["risk_score"] = (df.pause_norm*w1 + df.count_norm*w2 + df.iso_norm*w3)*100
    df["risk_score"] = df.risk_score.clip(0,100).round(1)
    return df

# --- PDF Generation ------------------------------------------------------

def make_detailed_pdf(df: pd.DataFrame, fig) -> io.BytesIO:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w,h = letter

    # --- Page 1: Title & Overview ---
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, h-50, "MemoTag Cognitive Decline Analysis")
    c.setFont("Helvetica", 11)
    overview = [
        "This application leverages OpenAI Whisper for speech transcription and librosa for",
        "voice activity detection (VAD) to quantify speech pauses.",
        "Prolonged or frequent pauses may indicate early cognitive decline.",
        "",
        "Key Features:",
        "- Automated transcription with Whisper",
        "- Pause detection via librosa.effects.split",
        "- Composite risk scoring using clustering & anomaly detection",
        "",
        "Applications: clinical screening, longitudinal monitoring, research."
    ]
    text = c.beginText(40, h-90)
    for line in overview:
        text.textLine(line)
    c.drawText(text)
    c.showPage()

    # --- Page 2: Methodology ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h-50, "Methodology")
    c.setFont("Helvetica", 11)
    steps = [
        "1. Transcription:",
        "   • Whisper ASR generates text transcript and word count.",
        "2. Voice Activity Detection:",
        "   • librosa.effects.split segments speech vs. silence.",
        "   • Compute total pause time & number of pauses.",
        "3. Feature Extraction:",
        "   • Optional MFCC statistics (13 coefficients).",
        "4. Scoring:",
        "   • DBSCAN clusters pause metrics.",
        "   • Isolation Forest yields anomaly score.",
        "   • Normalize & weight to 0–100 composite risk score."
    ]
    text = c.beginText(40, h-90)
    for line in steps:
        text.textLine(line)
    c.drawText(text)
    c.showPage()

    # --- Page 3: Results & Chart ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h-50, "Results Summary")
    # draw table
    row = df.iloc[0]
    results = [
        ("Total Duration (s)", f"{row.total_duration:.1f}"),
        ("Speech Rate (w/s)", f"{len(row.transcript.split())/row.total_duration:.2f}"),
        ("Total Pause (s)",      f"{row.total_pause:.1f}"),
        ("Number of Pauses",     f"{int(row.num_pauses)}"),
        ("Risk Score (%)",       f"{row.risk_score:.1f}")
    ]
    c.setFont("Helvetica", 11)
    y = h-90
    for label, val in results:
        c.drawString(60, y, f"{label}: {val}")
        y -= 20

    # embed scatter plot
    img = io.BytesIO()
    fig.savefig(img, format="PNG", bbox_inches="tight")
    img.seek(0)
    c.drawImage(ImageReader(img), 40, y-260, width=500, preserveAspectRatio=True)
    c.showPage()

    # --- Page 4: Glossary & Future Work ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h-50, "Glossary & Future Work")
    c.setFont("Helvetica", 11)
    glossary = [
        "Glossary:",
        "- VAD: Voice Activity Detection (speech vs. silence).",
        "- MFCC: Mel-frequency cepstral coefficients (spectral features).",
        "- DBSCAN: Density-based clustering algorithm.",
        "- Isolation Forest: Anomaly detection via random tree isolation.",
        "",
        "Future Work:",
        "• Validate thresholds on clinical datasets.",
        "• Integrate prosodic features (pitch, jitter, shimmer).",
        "• Longitudinal tracking dashboard.",
        "• REST API deployment for clinical use."
    ]
    text = c.beginText(40, h-90)
    for line in glossary:
        text.textLine(line)
    c.drawText(text)

    c.save()
    buf.seek(0)
    return buf

# --- App UI --------------------------------------------------------------

st.set_page_config(page_title="MemoTag Cognitive Decline", layout="wide")
st.title("📋 MemoTag Cognitive Decline Detection")

# Sidebar
st.sidebar.header("Settings")
language   = st.sidebar.selectbox("Transcription Language", ["en","hi","fr","es"])
model_name = st.sidebar.selectbox("Whisper Model", ["tiny","base","small","medium","large"])
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()

# File uploader
files = st.file_uploader("Upload audio (wav/mp3/m4a)", type=["wav","mp3","m4a"], accept_multiple_files=True)
if not files:
    st.info("Upload at least one audio file to begin."); st.stop()

audio_bytes = files[0].read()
records = [ extract_features(audio_bytes, language, model_name) ]
df = pd.DataFrame(records)
df = score_df(df)

# Playback & transcript
st.subheader("🔊 Playback & Transcript")
st.audio(audio_bytes)
st.markdown(f"**Transcript:** {df.transcript.iloc[0]}")

# Summary table
summary = df[["filename","total_duration","total_pause","num_pauses","risk_score"]].copy()
summary.columns = ["File","Duration (s)","Pause (s)","Pauses Count","Risk (%)"]
st.subheader("🔍 Extracted Features & Risk Scores")
st.dataframe(summary, use_container_width=True)

# Scatter plot
st.subheader("🗺️ Pause vs. Risk Score")
fig, ax = plt.subplots()
ax.scatter(df.total_pause, df.risk_score, s=60, edgecolor="k", alpha=0.7)
ax.set_xlabel("Total Pause Duration (s)")
ax.set_ylabel("Risk Score (0–100)")
st.pyplot(fig)

# Downloads
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "cognitive_report.csv", "text/csv")

pdf_buf = make_detailed_pdf(df, fig)
st.download_button("Download Detailed PDF Report", pdf_buf, "cognitive_report.pdf", "application/pdf")

# Inline PDF preview (optional)
b64 = base64.b64encode(pdf_buf.getvalue()).decode("utf-8")
iframe = f"""<iframe src="data:application/pdf;base64,{b64}" width="100%" height="350px"></iframe>"""
st.markdown(iframe, unsafe_allow_html=True)

# Executive summary
st.subheader("📝 Executive Summary")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"- **Total Pause:** {df.total_pause.iloc[0]:.1f}s")
    st.markdown(f"- **Pauses Count:** {int(df.num_pauses.iloc[0])}")
with c2:
    st.markdown("**Cognitive Risk Score**")
    st.markdown(f"<h1 style='color:#d6336c'>{df.risk_score.iloc[0]:.1f}%</h1>", unsafe_allow_html=True)

# Detailed Analysis in‑app
st.subheader("📑 Detailed Analysis Report")
report_md = """
### 1. Overview
This tool uses state‑of‑the‑art speech transcription (OpenAI Whisper) and audio analysis (librosa)  
to detect and quantify pauses in speech—potential early markers of cognitive decline.

### 2. Methodology
1. **Transcription**: Whisper ASR → text + word counts  
2. **Pause Detection**: librosa.effects.split → total pause duration & number of pauses  
3. **Features**: Optional MFCC statistics for spectral analysis  
4. **Scoring**: Combine normalized pause metrics with DBSCAN + Isolation Forest → 0–100 risk score  

### 3. Results
- **Total Duration**: `{:.1f}` s  
- **Speech Rate**: `{:.2f}` w/s  
- **Total Pause**: `{:.1f}` s  
- **Pauses Count**: `{}`  
- **Risk Score**: `{:.1f}` %

### 4. Glossary & Future Work
- **VAD**: Voice Activity Detection  
- **MFCC**: Mel‑frequency cepstral coefficients  
- **DBSCAN**: Density‑based clustering  
- **Isolation Forest**: Tree‑based anomaly detection  

**Future Work**  
- Clinical validation, prosodic features, longitudinal dashboards, API deployment.
""".format(
    df.total_duration.iloc[0],
    len(df.transcript.iloc[0].split())/df.total_duration.iloc[0],
    df.total_pause.iloc[0],
    int(df.num_pauses.iloc[0]),
    df.risk_score.iloc[0]
)
st.markdown(report_md)
