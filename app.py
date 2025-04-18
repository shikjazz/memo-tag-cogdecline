import warnings
warnings.filterwarnings("ignore")

import io
import tempfile
import base64
import textwrap

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
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

# --- Page config ---
st.set_page_config(page_title="MemoTag Cognitive Decline", layout="wide")

# --- Whisper model cache ---
@st.cache_resource(show_spinner=False)
def load_whisper(model_name="base"):
    return whisper.load_model(model_name)

# --- Feature extraction (no parselmouth) ---
@st.cache_data(show_spinner=False)
def extract_features(audio_bytes: bytes, language: str, model_name: str):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        seg.export(tmp.name, format="wav")
        path = tmp.name

    # 1) ASR
    model     = load_whisper(model_name)
    res       = model.transcribe(path, language=language)
    transcript = res["text"].strip()
    words     = transcript.split()

    # 2) VAD via librosa
    y, sr     = librosa.load(path, sr=None, mono=True)
    intervals = librosa.effects.split(y, top_db=25)
    speech_dur  = sum((e - s) for s, e in intervals) / sr
    total_dur   = len(y) / sr
    total_pause = total_dur - speech_dur
    num_pauses  = max(len(intervals) - 1, 0)
    speech_rate = len(words) / speech_dur if speech_dur > 0 else 0.0

    # 3) MFCC stats
    mfcc          = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means    = mfcc.mean(axis=1)
    mfcc_stds     = mfcc.std(axis=1)

    feats = {
        "filename":       tmp.name.split("/")[-1],
        "tmp_path":       path,
        "transcript":     transcript,
        "total_duration": total_dur,
        "speech_dur":     speech_dur,
        "total_pause":    total_pause,
        "num_pauses":     num_pauses,
        "speech_rate":    speech_rate,
    }
    for i in range(13):
        feats[f"mfcc_{i+1}_mean"] = float(mfcc_means[i])
        feats[f"mfcc_{i+1}_std"]  = float(mfcc_stds[i])
    return feats

# --- Risk scoring ---
@st.cache_data(show_spinner=False)
def score_df(df: pd.DataFrame) -> pd.DataFrame:
    X   = df[["total_pause", "num_pauses"]]
    db  = DBSCAN(eps=0.5, min_samples=2).fit(X)
    iso = IsolationForest(contamination=0.1).fit(X)
    df["dbscan_label"] = db.labels_
    df["iso_score"]    = iso.decision_function(X)

    pmax, nmax, imax = df.total_pause.max(), df.num_pauses.max(), df.iso_score.max()
    df["pause_norm"] = df.total_pause.apply(lambda v: v/pmax if pmax>0 else 0)
    df["count_norm"] = df.num_pauses.apply(lambda v: v/nmax if nmax>0 else 0)
    df["iso_norm"]   = df.iso_score.apply(lambda v: 1-(v/imax) if imax>0 else 0).fillna(0)

    w1, w2, w3 = 0.5, 0.3, 0.2
    df["risk_score"] = (df.pause_norm*w1 + df.count_norm*w2 + df.iso_norm*w3) * 100
    df["risk_score"] = df.risk_score.clip(0,100).round(1)
    return df

# --- PDF report builder ---
def make_pdf(df: pd.DataFrame, fig_hist, fig_scatter, risk_thresh:int) -> io.BytesIO:
    buf = io.BytesIO()
    c   = canvas.Canvas(buf, pagesize=letter)
    w,h = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h-40, "MemoTag Cognitive Decline Analysis Report")

    # Project Overview
    text = c.beginText(40, h-70)
    text.setFont("Helvetica", 10)
    overview = (
        "This report summarizes the analysis of speech pauses extracted "
        "from the uploaded audio. By leveraging OpenAI's Whisper for "
        "automatic speech recognition and signal processing via librosa, "
        "we quantify pausing patterns and compute a composite risk score "
        "indicative of potential cognitive decline."
    )
    for line in textwrap.wrap(overview, width=90):
        text.textLine(line)
    text.textLine("")
    c.drawText(text)

    # Key Metrics Table
    row = df.iloc[0]
    metrics = [
        ("Total Duration (s)", f"{row.total_duration:.1f}"),
        ("Speech Duration (s)", f"{row.speech_dur:.1f}"),
        ("Total Pause (s)",     f"{row.total_pause:.1f}"),
        ("# Pauses",            f"{int(row.num_pauses)}"),
        ("Speech Rate (w/s)",   f"{row.speech_rate:.2f}"),
        ("Risk Score (%)",      f"{row.risk_score:.1f}"),
        ("Category",            "High Risk" if row.risk_score>=risk_thresh else
                                "Medium Risk" if row.risk_score>=risk_thresh/2 else
                                "Low Risk"),
    ]
    y0 = h-180
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y0, "Key Metrics:")
    c.setFont("Helvetica", 10)
    for i,(label,val) in enumerate(metrics):
        c.drawString(60, y0 - 15*(i+1), f"{label}: {val}")

    # Embed Histogram
    img1 = io.BytesIO()
    fig_hist.savefig(img1, format="PNG", bbox_inches="tight")
    img1.seek(0)
    c.drawImage(ImageReader(img1), 40, y0-250, width=3.0*inch, height=2.0*inch)

    # Embed Scatter
    img2 = io.BytesIO()
    fig_scatter.savefig(img2, format="PNG", bbox_inches="tight")
    img2.seek(0)
    c.drawImage(ImageReader(img2), 340, y0-250, width=3.0*inch, height=2.0*inch)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf

# --- Sidebar settings ---
st.sidebar.header("Settings")
language    = st.sidebar.selectbox("Transcription Language", ["en","hi","fr","es"], index=0)
model_name  = st.sidebar.selectbox("Whisper Model",       ["tiny","base","small","medium","large"], index=1)
risk_thresh = st.sidebar.slider("High‑risk threshold (%)", 0, 100, 70)
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()

# --- Main UI ---
st.title("📋 MemoTag Cognitive Decline Detection")
files = st.file_uploader("Upload audio (wav/mp3/m4a)",
                         type=["wav","mp3","m4a"],
                         accept_multiple_files=True)
if not files:
    st.info("Upload at least one audio file to begin.")
    st.stop()

audio_bytes_list = [f.read() for f in files]
records = []
for b in audio_bytes_list:
    with st.spinner("Extracting features…"):
        records.append(extract_features(b, language, model_name))

df = pd.DataFrame(records)
df = score_df(df)

# KPI cards
st.subheader("🏷️ Key Metrics")
k1,k2,k3 = st.columns(3)
k1.metric("Total Duration (s)", f"{df.total_duration.iloc[0]:.1f}")
k2.metric("Speech Rate (w/s)",    f"{df.speech_rate.iloc[0]:.2f}")
k3.metric("Risk Score (%)",        f"{df.risk_score.iloc[0]:.1f}")

# Playback & transcript
st.subheader("🔊 Playback & Transcript")
st.audio(audio_bytes_list[0])
st.markdown(f"**Transcript:** {df.transcript.iloc[0]}")

# Clean summary table
summary = df[[
    "filename","total_duration","speech_dur",
    "total_pause","num_pauses","speech_rate","risk_score"
]].copy()
summary.columns = [
    "File","Total (s)","Speech (s)",
    "Pause (s)","# Pauses","Speech Rate (w/s)","Risk (%)"
]
st.subheader("🔍 Extracted Features & Risk Scores")
st.dataframe(summary, use_container_width=True)

# Pause‑length histogram
st.subheader("📊 Pause‑Length Distribution")
path       = df.tmp_path.iloc[0]
y, sr      = librosa.load(path, sr=None, mono=True)
intervals  = librosa.effects.split(y, top_db=25)
pause_lengths = [
    (intervals[i][0] - intervals[i-1][1]) / sr
    for i in range(1, len(intervals))
]
fig_hist, axh = plt.subplots(figsize=(6,3))
axh.hist(pause_lengths, bins=20, edgecolor="k", alpha=0.7)
axh.set_xlabel("Pause Length (s)")
axh.set_ylabel("Count")
st.pyplot(fig_hist)

# Pause vs Risk scatter
st.subheader("🗺️ Pause vs. Risk Score")
fig_sc, axsc = plt.subplots(figsize=(6,4))
colors = ["red" if r>=risk_thresh else "green" for r in df.risk_score]
axsc.scatter(df.total_pause, df.risk_score, s=80, c=colors, edgecolor="k", alpha=0.8)
axsc.set_xlabel("Total Pause Duration (s)")
axsc.set_ylabel("Risk Score (%)")
axsc.axhline(risk_thresh, color="gray", linestyle="--", linewidth=1)
st.pyplot(fig_sc)

# Detailed Analysis Report section
st.subheader("📑 Detailed Analysis Report")
st.markdown(
    """
**Project Overview**  
This analysis uses Whisper to transcribe your audio and librosa to detect speech‐pause intervals.  
By clustering and applying an isolation forest, we derive a composite *cognitive risk score* that can highlight potential signs of slowed speech or increased pausing—markers often associated with cognitive decline.

**Interpretation**  
- **High Pause Duration** and **many pauses** → higher risk score  
- **Speech Rate** (words per second) below typical conversational norms may also indicate slowed cognition  
- You can adjust the high‐risk threshold in the sidebar to see how your score category (Low/Medium/High) shifts.
"""  
)

# Downloads
csv     = df.to_csv(index=False).encode("utf-8")
pdf_buf = make_pdf(df, fig_hist, fig_sc, risk_thresh)

st.download_button("Download CSV", csv, "cognitive_report.csv", "text/csv")
st.download_button("Download PDF Report", pdf_buf, "cognitive_report.pdf", "application/pdf")

# Executive summary
st.subheader("📝 Executive Summary")
cat = (
    "High Risk"   if df.risk_score.iloc[0] >= risk_thresh else
    "Medium Risk" if df.risk_score.iloc[0] >= (risk_thresh/2) else
    "Low Risk"
)
c1,c2 = st.columns(2)
with c1:
    st.markdown(f"- **Total Pause:** {df.total_pause.iloc[0]:.1f}s")
    st.markdown(f"- **# Pauses:** {df.num_pauses.iloc[0]}")
with c2:
    st.markdown("**Cognitive Risk Score**")
    st.markdown(f"<h1 style='color:#d6336c'>{df.risk_score.iloc[0]:.1f}%</h1>",
                unsafe_allow_html=True)
    st.markdown(f"**Category:** {cat}")

st.markdown("""
**Next Steps**  
1. Validate thresholds with clinicians  
2. Deploy as a REST API endpoint  
3. Integrate longitudinal tracking dashboard  
4. Optimize Whisper model for CPU/GPU  
""")
