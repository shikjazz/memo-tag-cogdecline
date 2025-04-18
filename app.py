import warnings
warnings.filterwarnings("ignore")

import io
import tempfile

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

# Page config
st.set_page_config(page_title="MemoTag Cognitive Decline", layout="wide")

# Load Whisper model once
@st.cache_resource(show_spinner=False)
def load_whisper(model_name="base"):
    return whisper.load_model(model_name)

# Feature extraction with caching (no parselmouth)
@st.cache_data(show_spinner=False)
def extract_features(audio_bytes: bytes, language: str, model_name: str):
    # save to a temp .wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        seg.export(tmp.name, format="wav")
        path = tmp.name

    # ASR with Whisper
    model = load_whisper(model_name)
    res = model.transcribe(path, language=language)
    transcript = res["text"].strip()

    # Pause detection & MFCCs
    y, sr = librosa.load(path, sr=None, mono=True)
    intervals = librosa.effects.split(y, top_db=25)
    speech_dur = sum((e - s) for s, e in intervals) / sr
    total_dur = len(y) / sr
    total_pause = total_dur - speech_dur
    num_pauses = max(len(intervals) - 1, 0)

    # MFCC stats just in case
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfcc.mean(axis=1)
    mfcc_stds = mfcc.std(axis=1)

    feats = {
        "filename": tmp.name.split("/")[-1],
        "transcript": transcript,
        "total_duration": total_dur,
        "total_pause": total_pause,
        "num_pauses": num_pauses,
    }
    # add mfcc stats (optional)
    for i in range(13):
        feats[f"mfcc_{i+1}_mean"] = float(mfcc_means[i])
        feats[f"mfcc_{i+1}_std"] = float(mfcc_stds[i])
    return feats

# Clustering + risk scoring (pause‐only)
@st.cache_data(show_spinner=False)
def score_df(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["total_pause", "num_pauses"]]
    db = DBSCAN(eps=0.5, min_samples=2).fit(X)
    iso = IsolationForest(contamination=0.1).fit(X)
    df["dbscan_label"] = db.labels_
    df["iso_score"] = iso.decision_function(X)

    # weighted risk composite (0–100)
    w_pause, w_pauses, w_iso = 0.5, 0.3, 0.2
    r = (
        (df.total_pause / df.total_pause.max()) * w_pause
        + (df.num_pauses / df.num_pauses.max()) * w_pauses
        + (1 - (df.iso_score / df.iso_score.max())) * w_iso
    )
    df["risk_score"] = (r * 100).clip(0, 100).round(1)
    return df

# Build a one‑page PDF report
def make_pdf(df: pd.DataFrame, fig) -> io.BytesIO:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, h - 30, "MemoTag Cognitive Decline Report")

    text = c.beginText(30, h - 60)
    text.setFont("Helvetica", 10)
    row = df.iloc[0]
    for k in ["filename", "total_pause", "num_pauses", "risk_score"]:
        text.textLine(f"{k}: {row[k]}")
    c.drawText(text)

    # embed chart
    img = io.BytesIO()
    fig.savefig(img, format="PNG", bbox_inches="tight")
    img.seek(0)
    c.drawImage(ImageReader(img), 30, h - 300, width=550, preserveAspectRatio=True)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf

# --- UI ---
st.title("📋 MemoTag Cognitive Decline Detection")

# Sidebar settings
st.sidebar.header("Settings")
language = st.sidebar.selectbox("Transcription Language", ["en", "hi", "fr", "es"], index=0)
model_name = st.sidebar.selectbox("Whisper Model", ["tiny", "base", "small", "medium", "large"], index=1)
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()

# File uploader
files = st.file_uploader("Upload audio (wav/mp3/m4a)", type=["wav", "mp3", "m4a"], accept_multiple_files=True)
if not files:
    st.info("Upload at least one audio file to begin.")
    st.stop()

# Extract & score
records = []
for f in files:
    with st.spinner(f"Processing {f.name}…"):
        feats = extract_features(f.read(), language, model_name)
        feats["filename"] = f.name
        records.append(feats)

df = pd.DataFrame(records)
df = score_df(df)

# Audio playback & description
st.subheader("🔊 Audio Playback & Description")
st.audio(files[0].read(), format="audio/wav")
st.markdown("**Verbal Description (Transcript):**")
st.write(df.loc[0, "transcript"])

# Transposed feature table: metrics ↓, files →
st.subheader("🔍 Extracted Features & Scores")
wide_df = df.set_index("filename").T
fmt = {
    "total_duration": "{:.2f}",
    "total_pause":    "{:.2f}",
    "num_pauses":     "{:.0f}",
    "risk_score":     "{:.1f}"
}
st.dataframe(wide_df.style.format(fmt), use_container_width=True)

# Scatter plot
st.subheader("🗺️ Pause vs. Risk Score")
fig, ax = plt.subplots()
ax.scatter(df.total_pause, df.risk_score, s=60, edgecolor="k", alpha=0.7)
ax.set_xlabel("Total Pause Duration (s)")
ax.set_ylabel("Risk Score (0–100)")
st.pyplot(fig)

# CSV download
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "cognitive_report.csv", "text/csv")

# PDF download
pdf_buf = make_pdf(df, fig)
st.download_button("Download PDF Report", pdf_buf, "cognitive_report.pdf", "application/pdf")

# Executive summary
st.subheader("📝 Executive Summary")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"- **Total Pause:** {df.total_pause.iloc[0]:.1f}s")
    st.markdown(f"- **Num Pauses:** {df.num_pauses.iloc[0]}")
with c2:
    st.markdown("**Cognitive Risk Score**")
    st.markdown(f"<h1 style='color:#d6336c'>{df.risk_score.iloc[0]}</h1>", unsafe_allow_html=True)

st.markdown(
    """
**Next Steps**  
1. Validate thresholds with clinicians  
2. Deploy as a REST API endpoint  
3. Integrate longitudinal tracking dashboard  
4. Optimize Whisper model for CPU/GPU  
"""
)
