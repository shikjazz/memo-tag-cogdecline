import warnings
warnings.filterwarnings("ignore")

import io
import base64
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

st.set_page_config(page_title="MemoTag Cognitive Decline", layout="wide")

# --- Helpers ---
@st.cache_resource
def load_whisper(model_name: str):
    return whisper.load_model(model_name)

@st.cache_data
def extract_features(audio_bytes: bytes, language: str, model_name: str):
    # write to temp wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        seg.export(tmp.name, format="wav")
        path = tmp.name

    # ASR
    model = load_whisper(model_name)
    res = model.transcribe(path, language=language)
    transcript = res["text"].strip()

    # pause analysis
    y, sr = librosa.load(path, sr=None, mono=True)
    intervals = librosa.effects.split(y, top_db=25)
    speech = sum((e-s) for s,e in intervals)/sr
    total = len(y)/sr
    pause = total - speech
    n_pause = max(len(intervals)-1, 0)

    return {
        "filename": path.split("/")[-1],
        "transcript": transcript,
        "total_duration": total,
        "total_pause": pause,
        "num_pauses": n_pause
    }

@st.cache_data
def score_df(df: pd.DataFrame):
    X = df[["total_pause","num_pauses"]].fillna(0)
    db = DBSCAN(eps=0.5, min_samples=2).fit(X)
    iso = IsolationForest(contamination=0.1).fit(X)
    df["dbscan_label"] = db.labels_
    df["iso_score"] = iso.decision_function(X)

    # safe norms
    pmax = df.total_pause.max() or 1
    nmax = df.num_pauses .max() or 1
    imax = df.iso_score  .max() or 1

    df["pause_norm"] = df.total_pause / pmax
    df["count_norm"] = df.num_pauses  / nmax
    df["iso_norm"]   = 1 - (df.iso_score/imax)

    w_p, w_n, w_i = 0.5, 0.3, 0.2
    df["risk_score"] = (
        df.pause_norm * w_p +
        df.count_norm * w_n +
        df.iso_norm   * w_i
    ) * 100
    df["risk_score"] = df.risk_score.clip(0,100).round(1)
    return df

def make_pdf(df: pd.DataFrame, fig) -> io.BytesIO:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, h-30, "MemoTag Cognitive Decline Report")

    # text
    text = c.beginText(30, h-60)
    text.setFont("Helvetica", 10)
    row = df.iloc[0]
    for k in ["filename","total_pause","num_pauses","risk_score"]:
        text.textLine(f"{k}: {row[k]}")
    c.drawText(text)

    # chart
    img = io.BytesIO()
    fig.savefig(img, format="PNG", bbox_inches="tight")
    img.seek(0)
    c.drawImage(ImageReader(img), 30, h-300, width=550, preserveAspectRatio=True)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf

# --- UI ---
st.title("📋 MemoTag Cognitive Decline Detection")

st.sidebar.header("Settings")
lang = st.sidebar.selectbox("Language", ["en","hi","fr","es"])
model = st.sidebar.selectbox("Whisper Model", ["tiny","base","small","medium","large"])
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear(); st.cache_resource.clear()

files = st.file_uploader("Upload audio", type=["wav","mp3","m4a"], accept_multiple_files=True)
if not files:
    st.info("Please upload at least one audio file.")
    st.stop()

# read & process
data = [f.read() for f in files]
records = [extract_features(b, lang, model) for b in data]
df = pd.DataFrame(records)
df = score_df(df)

# audio + transcript
st.subheader("🔊 Playback & Transcript")
st.audio(data[0])
st.markdown(f"**Transcript:** {df.transcript.iloc[0]}")

# table
st.subheader("🔍 Extracted Features & Risk Scores")
st.table(df[[
    "filename","total_duration","total_pause","num_pauses","risk_score"
]].rename(columns={
    "filename":"File",
    "total_duration":"Duration (s)",
    "total_pause":"Pause (s)",
    "num_pauses":"# Pauses",
    "risk_score":"Risk (%)"
}))

# scatter
st.subheader("🗺️ Pause vs Risk")
fig, ax = plt.subplots()
ax.scatter(df.total_pause, df.risk_score, s=80, edgecolor="k", alpha=0.7)
ax.set_xlabel("Pause (s)")
ax.set_ylabel("Risk (%)")
st.pyplot(fig)

# downloads
csv = df.to_csv(index=False).encode()
st.download_button("Download CSV", csv, "report.csv", "text/csv")

pdf = make_pdf(df, fig)
b64 = base64.b64encode(pdf.read()).decode("utf-8")
st.markdown(
    f'<iframe src="data:application/pdf;base64,{b64}" '
    'width="100%" height="600px"></iframe>',
    unsafe_allow_html=True
)
st.download_button("Download PDF Report", pdf, "report.pdf", "application/pdf")

# summary
st.subheader("📝 Executive Summary")
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"- **Pause:** {df.total_pause.iloc[0]:.1f}s")
    st.markdown(f"- **# Pauses:** {df.num_pauses.iloc[0]}")
with col2:
    st.markdown(f"<h1 style='color:#d6336c'>{df.risk_score.iloc[0]}%</h1>", unsafe_allow_html=True)
