import warnings
warnings.filterwarnings("ignore")

import io
import tempfile
import textwrap
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
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from st_aggrid import AgGrid, GridOptionsBuilder

# --- Page config ---
st.set_page_config(page_title="MemoTag Cognitive Decline", layout="wide")

# --- Whisper model cache ---
@st.cache_resource(show_spinner=False)
def load_whisper(model_name="base"):
    return whisper.load_model(model_name)

# --- Feature extraction ---
@st.cache_data(show_spinner=False)
def extract_features(audio_bytes: bytes, language: str, model_name: str):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        seg.export(tmp.name, format="wav")
        path = tmp.name

    # Transcription
    model      = load_whisper(model_name)
    res        = model.transcribe(path, language=language)
    transcript = res["text"].strip()
    words      = transcript.split()

    # Pause detection
    y, sr     = librosa.load(path, sr=None, mono=True)
    intervals = librosa.effects.split(y, top_db=25)
    speech_dur  = sum((e - s) for s, e in intervals) / sr
    total_dur   = len(y) / sr
    total_pause = total_dur - speech_dur
    num_pauses  = max(len(intervals) - 1, 0)
    speech_rate = len(words) / speech_dur if speech_dur > 0 else 0.0

    # MFCC summary
    mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfcc.mean(axis=1)
    mfcc_stds  = mfcc.std(axis=1)

    feats = {
        "filename":       tmp.name.split("/")[-1],
        "tmp_path":       path,
        "transcript":     transcript,
        "total_duration": total_dur,
        "speech_dur":     speech_dur,
        "total_pause":    total_pause,
        "num_pauses":     num_pauses,
        "speech_rate":    speech_rate
    }
    for i in range(13):
        feats[f"mfcc_{i+1}_mean"] = float(mfcc_means[i])
        feats[f"mfcc_{i+1}_std"]  = float(mfcc_stds[i])
    return feats

# --- Scoring ---
@st.cache_data(show_spinner=False)
def score_df(df: pd.DataFrame) -> pd.DataFrame:
    X   = df[["total_pause","num_pauses"]]
    db  = DBSCAN(eps=0.5, min_samples=2).fit(X)
    iso = IsolationForest(contamination=0.1).fit(X)
    df["dbscan_label"] = db.labels_
    df["iso_score"]    = iso.decision_function(X)

    pmax, nmax, imax = df.total_pause.max(), df.num_pauses.max(), df.iso_score.max()
    df["pause_norm"] = df.total_pause.apply(lambda v: v/pmax if pmax>0 else 0)
    df["count_norm"] = df.num_pauses.apply(lambda v: v/nmax if nmax>0 else 0)
    df["iso_norm"]   = df.iso_score.apply(lambda v:1-(v/imax) if imax>0 else 0).fillna(0)

    w1,w2,w3 = 0.5,0.3,0.2
    df["risk_score"] = (df.pause_norm*w1 + df.count_norm*w2 + df.iso_norm*w3)*100
    df["risk_score"] = df.risk_score.clip(0,100).round(1)
    return df

# --- Multi‑page PDF builder ---
def make_pdf(df, fig_hist, fig_scatter, risk_thresh):
    buf = io.BytesIO()
    c   = canvas.Canvas(buf, pagesize=letter)
    w,h = letter

    # --- Page 1: Title & Overview ---
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40,h-40,"MemoTag Cognitive Decline Analysis Report")

    text = c.beginText(40,h-80)
    text.setFont("Helvetica", 11)
    overview = (
        "This application uses OpenAI Whisper for speech transcription "
        "and librosa for audio analysis to detect speech pauses, "
        "which are then combined into a composite risk score. "
        "Elevated pause durations and frequency can be early indicators "
        "of cognitive decline."
    )
    for line in textwrap.wrap(overview,100):
        text.textLine(line)
    c.drawText(text)
    c.showPage()

    # --- Page 2: Methodology ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40,h-40,"Methodology")
    text = c.beginText(40,h-80)
    text.setFont("Helvetica",10)
    for section,body in [
        ("1. Transcription", 
         "Audio is transcribed with Whisper, yielding a text transcript and word count."),
        ("2. VAD & Pause Detection",
         "We segment speech vs. silence using librosa.effects.split and compute total pause time, number of pauses."),
        ("3. Feature Extraction",
         "MFCC-based spectral features are extracted but not used directly in the risk score."),
        ("4. Scoring",
         "A DBSCAN cluster label and Isolation Forest outlier score normalize pause metrics into a 0-100 risk score.")
    ]:
        text.textLine(f"{section}:")
        for line in textwrap.wrap(body,90):
            text.textLine("   " + line)
        text.textLine("")
    c.drawText(text)
    c.showPage()

    # --- Page 3: Results & Figures ---
    row = df.iloc[0]
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40,h-40,"Results & Figures")
    c.setFont("Helvetica",11)
    metrics = [
        ("Total Duration (s)", f"{row.total_duration:.1f}"),
        ("Speech Rate (w/s)", f"{row.speech_rate:.2f}"),
        ("Total Pause (s)", f"{row.total_pause:.1f}"),
        ("# Pauses", f"{int(row.num_pauses)}"),
        ("Risk Score (%)", f"{row.risk_score:.1f}")
    ]
    for i,(label,val) in enumerate(metrics):
        c.drawString(40, h-80 - 18*i, f"{label}: {val}")

    # histogram
    img1 = io.BytesIO()
    fig_hist.savefig(img1, format="PNG", bbox_inches="tight")
    img1.seek(0)
    c.drawImage(ImageReader(img1), 300, h-300, width=3.0*inch, height=2.0*inch)

    # scatter
    img2 = io.BytesIO()
    fig_scatter.savefig(img2,format="PNG",bbox_inches="tight")
    img2.seek(0)
    c.drawImage(ImageReader(img2), 40, h-300, width=3.0*inch, height=2.0*inch)
    c.showPage()

    # --- Page 4: Glossary & Future Work ---
    c.setFont("Helvetica-Bold",16)
    c.drawString(40,h-40,"Glossary & Future Work")
    text = c.beginText(40,h-80)
    text.setFont("Helvetica",10)
    text.textLine("Glossary:")
    for term,definition in [
        ("VAD","Voice Activity Detection—separating speech vs. silence."),
        ("MFCC","Mel‑Frequency Cepstral Coefficients—spectral features."),
        ("DBSCAN","Density‑based clustering algorithm."),
        ("Isolation Forest","Anomaly detection via random tree isolation.")
    ]:
        text.textLine(f" • {term}: {definition}")
    text.textLine("")
    text.textLine("Future Work:")
    for fw in [
        "Validate risk thresholds against clinical datasets.",
        "Incorporate prosodic features (pitch, jitter, shimmer).",
        "Build longitudinal tracking over multiple sessions.",
        "Deploy REST API and web dashboard for clinicians."
    ]:
        text.textLine(f" • {fw}")
    c.drawText(text)
    c.showPage()

    c.save()
    buf.seek(0)
    return buf

# --- Sidebar settings ---
st.sidebar.header("Settings")
language    = st.sidebar.selectbox("Transcription Language", ["en","hi","fr","es"], index=0)
model_name  = st.sidebar.selectbox("Whisper Model", ["tiny","base","small","medium","large"], index=1)
risk_thresh = st.sidebar.slider("High‑risk threshold (%)", 0, 100, 70)
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()

# --- Main ---
st.title("📋 MemoTag Cognitive Decline Detection")
files = st.file_uploader("Upload audio (wav/mp3/m4a)", type=["wav","mp3","m4a"], accept_multiple_files=True)
if not files:
    st.info("Upload at least one file to begin.")
    st.stop()

audio_bytes_list = [f.read() for f in files]
records          = [extract_features(b,language,model_name) for b in audio_bytes_list]
df               = score_df(pd.DataFrame(records))

# KPI cards
st.subheader("🏷️ Key Metrics")
c1,c2,c3 = st.columns(3)
c1.metric("Duration (s)",  f"{df.total_duration.iloc[0]:.1f}")
c2.metric("Speech Rate",    f"{df.speech_rate.iloc[0]:.2f} w/s")
c3.metric("Risk Score (%)", f"{df.risk_score.iloc[0]:.1f}%")

# Playback & transcript
st.subheader("🔊 Playback & Transcript")
st.audio(audio_bytes_list[0])
st.write(df.transcript.iloc[0])

# Interactive summary table via AgGrid
summary = df[[
    "filename","total_duration","speech_dur",
    "total_pause","num_pauses","speech_rate","risk_score"
]].copy()
summary.columns = [
    "File","Total (s)","Spoken (s)",
    "Pause (s)","# Pauses","Rate (w/s)","Risk (%)"
]
st.subheader("🔍 Extracted Features & Risk Scores")
gb = GridOptionsBuilder.from_dataframe(summary)
gb.configure_column("Risk (%)", cellStyle={
    "function": "params.value >= %d ? {'color':'red'} : {'color':'green'}" % risk_thresh
})
gridOpts = gb.build()
AgGrid(summary, gridOptions=gridOpts, enable_enterprise_modules=False)

# Histogram
st.subheader("📊 Pause Length Distribution")
fig_hist, axh = plt.subplots(figsize=(6,3))
y,sr          = librosa.load(df.tmp_path.iloc[0], sr=None, mono=True)
ints          = librosa.effects.split(y, top_db=25)
pause_lens    = [(ints[i][0]-ints[i-1][1])/sr for i in range(1,len(ints))]
axh.hist(pause_lens, bins=20, edgecolor="k", alpha=0.7)
axh.set_xlabel("Pause Length (s)")
axh.set_ylabel("Count")
st.pyplot(fig_hist)

# Scatter
st.subheader("🗺️ Pause vs. Risk Score")
fig_sc, axsc = plt.subplots(figsize=(6,4))
colors = ["red" if r>=risk_thresh else "green" for r in df.risk_score]
axsc.scatter(df.total_pause, df.risk_score, c=colors, edgecolor="k", s=80, alpha=0.8)
axsc.axhline(risk_thresh, color="gray", linestyle="--")
axsc.set_xlabel("Total Pause (s)")
axsc.set_ylabel("Risk (%)")
st.pyplot(fig_sc)

# In‑app Detailed Analysis
st.subheader("📑 Detailed Analysis Report")
with st.expander("Project Overview"):
    st.write(
        "This tool leverages Whisper for transcription and librosa for audio analysis "
        "to detect speech-pause patterns. Elevated pausing can indicate slowed cognition."
    )
with st.expander("Methodology"):
    st.write(
        "- **Speech‑to‑Text:** Whisper\n"
        "- **Pause Detection:** librosa.effects.split\n"
        "- **Feature Extraction:** pause metrics, MFCCs\n"
        "- **Scoring:** DBSCAN + IsolationForest → 0‑100 risk"
    )
with st.expander("Glossary"):
    st.write(
        "**VAD:** Voice Activity Detection\n"
        "**MFCC:** Mel‑Frequency Cepstral Coefficients\n"
        "**DBSCAN:** Density‑based clustering\n"
        "**IsolationForest:** Anomaly detection"
    )
with st.expander("Results & Figures"):
    st.write("See above histogram and scatter for pause distribution and risk mapping.")
with st.expander("Future Work"):
    st.write(
        "- Validate against clinical data\n"
        "- Add prosodic features (pitch/jitter)\n"
        "- Longitudinal tracking\n"
        "- Deploy REST API/dashboard"
    )

# Downloads
csv     = df.to_csv(index=False).encode("utf-8")
pdf_buf = make_pdf(df, fig_hist, fig_sc, risk_thresh)

st.download_button("Download CSV", csv, "cognitive_report.csv", "text/csv")
st.download_button("Download PDF Report", pdf_buf, "cognitive_report.pdf", "application/pdf")
