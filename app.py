import warnings
warnings.filterwarnings("ignore")

import io
import tempfile

import streamlit as st
import whisper
import librosa
import librosa.display
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

# Persist last‑used settings
if "weights" not in st.session_state:
    st.session_state.weights = {"w_pause":0.5, "w_count":0.3, "w_iso":0.2}
if "cluster" not in st.session_state:
    st.session_state.cluster = {"eps":0.5, "min_s":2, "cont":0.1}

# --- Sidebar: adjustable sliders & settings ---
st.sidebar.header("Settings")
language = st.sidebar.selectbox("Transcription Language", ["en","hi","fr","es"], index=0)
model_name = st.sidebar.selectbox("Whisper Model", ["tiny","base","small","medium","large"], index=1)

st.sidebar.subheader("Risk Weights & Clustering")
st.session_state.weights["w_pause"] = st.sidebar.slider("Pause weight", 0.0,1.0, st.session_state.weights["w_pause"], step=0.05)
st.session_state.weights["w_count"] = st.sidebar.slider("Pause count weight", 0.0,1.0, st.session_state.weights["w_count"], step=0.05)
st.session_state.weights["w_iso"]   = st.sidebar.slider("IsoForest weight", 0.0,1.0, st.session_state.weights["w_iso"], step=0.05)
st.session_state.cluster["eps"]     = st.sidebar.slider("DBSCAN eps", 0.1,2.0, st.session_state.cluster["eps"], step=0.1)
st.session_state.cluster["min_s"]   = st.sidebar.slider("DBSCAN min_samples", 1,10, st.session_state.cluster["min_s"], step=1)
st.session_state.cluster["cont"]    = st.sidebar.slider("IsoForest contamination", 0.01,0.5, st.session_state.cluster["cont"], step=0.01)

if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()

# Load Whisper model once
@st.cache_resource(show_spinner=False)
def load_whisper(model_name="base"):
    return whisper.load_model(model_name)

# Helper for WPM
def compute_wpm(transcript: str, dur: float) -> float:
    words = len(transcript.split())
    return words / (dur/60) if dur>0 else 0

# Feature extraction with caching (wrapped in try/except)
@st.cache_data(show_spinner=False)
def extract_features(audio_bytes: bytes, language: str, model_name: str):
    # save to temp .wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        seg.export(tmp.name, format="wav")
        path = tmp.name

    # ASR
    model = load_whisper(model_name)
    res = model.transcribe(path, language=language)
    transcript = res["text"].strip()

    # audio load
    y, sr = librosa.load(path, sr=None, mono=True)
    intervals = librosa.effects.split(y, top_db=25)
    speech_dur = sum((e-s) for s,e in intervals)/sr
    total_dur  = len(y)/sr
    total_pause= total_dur - speech_dur
    num_pauses = max(len(intervals)-1, 0)

    # mfcc stats
    mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfcc.mean(axis=1)
    mfcc_stds  = mfcc.std(axis=1)

    feats = {
        "filename":    tmp.name.split("/")[-1],
        "transcript":  transcript,
        "total_duration": total_dur,
        "total_pause":    total_pause,
        "num_pauses":     num_pauses,
        "wpm":            compute_wpm(transcript, total_dur)
    }
    for i in range(13):
        feats[f"mfcc_{i+1}_mean"] = float(mfcc_means[i])
        feats[f"mfcc_{i+1}_std"]  = float(mfcc_stds[i])
    return feats

# Scoring with interactive parameters
@st.cache_data(show_spinner=False)
def score_df(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["total_pause","num_pauses"]]
    db = DBSCAN(eps=st.session_state.cluster["eps"], min_samples=st.session_state.cluster["min_s"]).fit(X)
    iso= IsolationForest(contamination=st.session_state.cluster["cont"]).fit(X)
    df["dbscan_label"] = db.labels_
    df["iso_score"]    = iso.decision_function(X)

    # safe norms
    pmax = df.total_pause.max() or 1
    cmax = df.num_pauses.max()  or 1
    imax = df.iso_score.max()   or 1

    df["pause_norm"] = df.total_pause / pmax
    df["count_norm"] = df.num_pauses / cmax
    df["iso_norm"]   = 1 - (df.iso_score / imax)

    # weighted composite
    w = st.session_state.weights
    df["risk_score"] = ( df.pause_norm*w["w_pause"]
                       + df.count_norm*w["w_count"]
                       + df.iso_norm  *w["w_iso"]
                       ) * 100
    df["risk_score"] = df.risk_score.clip(0,100).round(1)
    return df

# PDF builder unchanged...

def make_pdf(df: pd.DataFrame, fig) -> io.BytesIO:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w,h = letter
    c.setFont("Helvetica-Bold",14)
    c.drawString(30,h-30,"MemoTag Cognitive Decline Report")
    text = c.beginText(30,h-60); text.setFont("Helvetica",10)
    row = df.iloc[0]
    for k in ["filename","total_pause","num_pauses","risk_score","wpm"]:
        text.textLine(f"{k}: {row[k]}")
    c.drawText(text)
    img = io.BytesIO()
    fig.savefig(img,format="PNG",bbox_inches="tight")
    img.seek(0)
    c.drawImage(ImageReader(img),30,h-320,width=550,preserveAspectRatio=True)
    c.showPage(); c.save(); buf.seek(0)
    return buf

# --- Main UI ---
st.title("📋 MemoTag Cognitive Decline Detection")

# Upload
files = st.file_uploader("Upload audio (wav/mp3/m4a)", type=["wav","mp3","m4a"], accept_multiple_files=True)
if not files:
    st.info("Upload at least one audio file to begin.")
    st.stop()

audio_bytes_list = [f.read() for f in files]
records = []
for idx, audio_bytes in enumerate(audio_bytes_list):
    try:
        feats = extract_features(audio_bytes, language, model_name)
        records.append(feats)
    except Exception as e:
        st.error(f"⚠️ Failed processing file #{idx+1}: {e}")

df = pd.DataFrame(records)
df = score_df(df)

# Audio + transcription
st.subheader("🔊 Audio Playback & Description")
st.audio(audio_bytes_list[0], format="audio/wav")
st.markdown("**Verbal Description (Transcript):**")
st.write(df.transcript.iloc[0])

# waveform & spectrogram
y, sr = librosa.load(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name, sr=None)  # re‑load first
fig_w, ax_w = plt.subplots()
ax_w.plot(y); ax_w.set_title("Waveform")
st.pyplot(fig_w)
fig_s, ax_s = plt.subplots()
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y))), sr=sr, ax=ax_s, y_axis='log', x_axis='time')
ax_s.set_title("Spectrogram")
st.pyplot(fig_s)

# Detailed table
st.subheader("🔍 Extracted Features & Scores")
st.dataframe(df, use_container_width=True)

# Per‑file expanders
for i, row in df.iterrows():
    with st.expander(f"{row.filename}  –  Risk: {row.risk_score}%"):
        st.markdown(f"- **Pause**: {row.total_pause:.1f}s  \n"
                    f"- **Pauses**: {row.num_pauses}  \n"
                    f"- **WPM**: {row.wpm:.1f}")
        st.audio(audio_bytes_list[i], format="audio/wav")

# Scatter & distributions
st.subheader("🗺️ Pause vs. Risk Score")
fig, ax = plt.subplots()
ax.scatter(df.total_pause, df.risk_score, s=60, edgecolor="k", alpha=0.7)
ax.set_xlabel("Total Pause Duration (s)")
ax.set_ylabel("Risk Score (0–100)")
st.pyplot(fig)

st.subheader("📊 Feature Distributions")
fig_h, axes = plt.subplots(1,3,figsize=(12,3))
axes[0].hist(df.total_pause, bins=10); axes[0].set_title("Pause")
axes[1].hist(df.num_pauses, bins=10); axes[1].set_title("Pause Count")
axes[2].hist(df.risk_score, bins=10); axes[2].set_title("Risk Score")
st.pyplot(fig_h)

# Downloads
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "cognitive_report.csv", "text/csv")
pdf_buf = make_pdf(df, fig)
st.download_button("Download PDF Report", pdf_buf, "cognitive_report.pdf", "application/pdf")

# Executive summary
st.subheader("📝 Executive Summary")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"- **Total Pause:** {df.total_pause.iloc[0]:.1f}s")
    st.markdown(f"- **Num Pauses:** {df.num_pauses.iloc[0]}")
    st.markdown(f"- **WPM:** {df.wpm.iloc[0]:.1f}")
with c2:
    st.markdown("**Cognitive Risk Score**")
    st.markdown(f"<h1 style='color:#d6336c'>{df.risk_score.iloc[0]}</h1>", unsafe_allow_html=True)

st.markdown("""
**Next Steps**  
1. Validate thresholds with clinicians  
2. Deploy as a REST API endpoint  
3. Integrate longitudinal tracking dashboard  
4. Optimize Whisper model for CPU/GPU  
""")
