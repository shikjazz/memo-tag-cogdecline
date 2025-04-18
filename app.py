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

# --- Page config ---
st.set_page_config(page_title="MemoTag Cognitive Decline", layout="wide")

# --- Whisper model cache ---
@st.cache_resource(show_spinner=False)
def load_whisper(model_name="base"):
    return whisper.load_model(model_name)

# --- Feature extraction (no parselmouth) ---
@st.cache_data(show_spinner=False)
def extract_features(audio_bytes: bytes, language: str, model_name: str):
    # write to temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        seg.export(tmp.name, format="wav")
        path = tmp.name

    # 1) ASR
    model = load_whisper(model_name)
    res   = model.transcribe(path, language=language)
    transcript = res["text"].strip()
    words = transcript.split()

    # 2) VAD via librosa
    y, sr = librosa.load(path, sr=None, mono=True)
    intervals = librosa.effects.split(y, top_db=25)
    speech_dur  = sum((e - s) for s, e in intervals) / sr
    total_dur   = len(y) / sr
    total_pause = total_dur - speech_dur
    num_pauses  = max(len(intervals) - 1, 0)
    speech_rate = len(words) / speech_dur if speech_dur > 0 else 0.0

    # 3) MFCC stats (optional)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfcc.mean(axis=1)
    mfcc_stds  = mfcc.std(axis=1)

    feats = {
        "filename":       tmp.name.split("/")[-1],
        "tmp_path":       path,                     # for histogram reuse
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
def make_pdf(df: pd.DataFrame, fig) -> io.BytesIO:
    buf = io.BytesIO()
    c   = canvas.Canvas(buf, pagesize=letter)
    w,h = letter
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, h-30, "MemoTag Cognitive Decline Report")

    text = c.beginText(30, h-60)
    text.setFont("Helvetica", 10)
    row = df.iloc[0]
    for k in ("filename","total_duration","speech_dur","total_pause","num_pauses","speech_rate","risk_score"):
        text.textLine(f"{k}: {row[k]}")
    c.drawText(text)

    img = io.BytesIO()
    fig.savefig(img, format="PNG", bbox_inches="tight")
    img.seek(0)
    c.drawImage(ImageReader(img), 30, h-350, width=550, preserveAspectRatio=True)

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
files = st.file_uploader("Upload audio (wav/mp3/m4a)", type=["wav","mp3","m4a"], accept_multiple_files=True)
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
k2.metric("Speech Rate (w/s)",     f"{df.speech_rate.iloc[0]:.2f}")
k3.metric("Risk Score (%)",         f"{df.risk_score.iloc[0]:.1f}")

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
# reload the same file
path = df.tmp_path.iloc[0]
y, sr = librosa.load(path, sr=None, mono=True)
intervals = librosa.effects.split(y, top_db=25)
# compute pauses between intervals
pause_lengths = [
    (intervals[i][0] - intervals[i-1][1]) / sr
    for i in range(1, len(intervals))
]
fig2, ax2 = plt.subplots(figsize=(6,3))
ax2.hist(pause_lengths, bins=20, edgecolor="k", alpha=0.7)
ax2.set_xlabel("Pause Length (s)")
ax2.set_ylabel("Count")
st.pyplot(fig2)

# Pause vs Risk scatter
st.subheader("🗺️ Pause vs. Risk Score")
fig, ax = plt.subplots(figsize=(6,4))
colors = ["red" if r>risk_thresh else "green" for r in df.risk_score]
ax.scatter(df.total_pause, df.risk_score, s=80, c=colors, edgecolor="k", alpha=0.8)
ax.set_xlabel("Total Pause Duration (s)")
ax.set_ylabel("Risk Score (%)")
ax.axhline(risk_thresh, color="gray", linestyle="--", linewidth=1)
st.pyplot(fig)

# Downloads & PDF preview
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "cognitive_report.csv", "text/csv")

pdf_buf = make_pdf(df, fig)
st.download_button("Download PDF Report", pdf_buf, "cognitive_report.pdf", "application/pdf")
b64 = base64.b64encode(pdf_buf.getvalue()).decode("utf-8")
iframe = f"""<iframe src="data:application/pdf;base64,{b64}" width="100%" height="300px"></iframe>"""
st.markdown(iframe, unsafe_allow_html=True)

# Executive summary + category
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
    st.markdown(f"<h1 style='color:#d6336c'>{df.risk_score.iloc[0]:.1f}%</h1>", unsafe_allow_html=True)
    st.markdown(f"**Category:** {cat}")

st.markdown("""
**Next Steps**  
1. Validate thresholds with clinicians  
2. Deploy as a REST API endpoint  
3. Integrate longitudinal tracking dashboard  
4. Optimize Whisper model for CPU/GPU  
""")
