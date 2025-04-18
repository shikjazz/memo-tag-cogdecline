import warnings
warnings.filterwarnings("ignore")

import io
import os
import json
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
import plotly.express as px
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from st_aggrid import AgGrid, GridOptionsBuilder

# --- Page config & theming ---
st.set_page_config(page_title="MemoTag Cognitive Decline", layout="wide", initial_sidebar_state="expanded")
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

st.sidebar.checkbox("Dark mode", value=st.session_state.dark_mode, on_change=toggle_theme)
if st.session_state.dark_mode:
    st.markdown("""<style>body { background-color: #303030; color: #EEE; }</style>""", unsafe_allow_html=True)

# --- Whisper model cache ---
@st.cache_resource(show_spinner=False)
def load_whisper(model_name="base"):
    return whisper.load_model(model_name)

# --- Feature extraction with prosody & error handling ---
@st.cache_data(show_spinner=False)
def extract_features(audio_bytes: bytes, language: str, model_name: str):
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
            seg.export(tmp.name, format="wav")
            path = tmp.name

        # Transcription
        model      = load_whisper(model_name)
        res        = model.transcribe(path, language=language)
        transcript = res["text"].strip()
        words      = transcript.split()

        # Audio load
        y, sr = librosa.load(path, sr=None, mono=True)

        # Voice Activity Detection & Pause metrics
        intervals   = librosa.effects.split(y, top_db=25)
        speech_dur  = sum((e - s) for s, e in intervals) / sr
        total_dur   = len(y) / sr
        total_pause = total_dur - speech_dur
        num_pauses  = max(len(intervals) - 1, 0)
        speech_rate = len(words) / speech_dur if speech_dur > 0 else 0.0

        # Prosodic: pitch & loudness
        f0, voiced_flag, _ = librosa.pyin(y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'))
        pitch_mean = np.nanmean(f0)
        pitch_var  = np.nanvar(f0)
        rms        = librosa.feature.rms(y=y).mean()

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
            "speech_rate":    speech_rate,
            "pitch_mean":     pitch_mean,
            "pitch_var":      pitch_var,
            "rms":            rms
        }
        for i in range(13):
            feats[f"mfcc_{i+1}_mean"] = float(mfcc_means[i])
            feats[f"mfcc_{i+1}_std"]  = float(mfcc_stds[i])
        return feats
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None

# --- Scoring ---
@st.cache_data(show_spinner=False)
def score_df(df: pd.DataFrame) -> pd.DataFrame:
    X   = df[["total_pause","num_pauses"]]
    db  = DBSCAN(eps=0.5, min_samples=2).fit(X)
    iso = IsolationForest(contamination=0.1).fit(X)
    df["dbscan_label"] = db.labels_
    df["iso_score"]    = iso.decision_function(X)
    pmax, nmax, imax   = df.total_pause.max(), df.num_pauses.max(), df.iso_score.max()
    df["pause_norm"] = df.total_pause.apply(lambda v: v/pmax if pmax>0 else 0)
    df["count_norm"] = df.num_pauses.apply(lambda v: v/nmax if nmax>0 else 0)
    df["iso_norm"]   = df.iso_score.apply(lambda v: 1-(v/imax) if imax>0 else 0).fillna(0)
    w1,w2,w3        = 0.5,0.3,0.2
    df["risk_score"] = (df.pause_norm*w1 + df.count_norm*w2 + df.iso_norm*w3)*100
    df["risk_score"] = df.risk_score.clip(0,100).round(1)
    return df

# --- Trend over sessions ---
def plot_trend(all_dfs):
    trend = pd.DataFrame({
        "session": list(range(1, len(all_dfs)+1)),
        "risk_score": [df.risk_score.iloc[0] for df in all_dfs]
    })
    fig = px.line(trend, x="session", y="risk_score", markers=True,
                  title="Risk Score Over Multiple Sessions")
    st.plotly_chart(fig, use_container_width=True)

# --- PDF builder (no change) ---
def make_pdf(df, fig_hist, fig_scatter, risk_thresh):
    # ... your existing multi‑page PDF builder here ...
    pass  # omitted for brevity

# --- Sidebar settings ---
st.sidebar.header("Settings")
language    = st.sidebar.selectbox("Transcription Language", ["en","hi","fr","es"], index=0)
model_name  = st.sidebar.selectbox("Whisper Model", ["tiny","base","small","medium","large"], index=1)
risk_thresh = st.sidebar.slider("High‑risk threshold (%)", 0, 100, 70)
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear(); st.cache_resource.clear()

# --- Main UI ---
st.title("📋 MemoTag Cognitive Decline Detection")

files = st.file_uploader(
    "Upload audio (wav/mp3/m4a)", type=["wav","mp3","m4a"],
    accept_multiple_files=True
)
if not files:
    st.info("👆 Upload at least one file to begin.")
    st.stop()

# Process all sessions
all_records = []
all_dfs     = []
for f in files:
    feats = extract_features(f.read(), language, model_name)
    if feats:
        all_records.append(feats)
df = pd.DataFrame(all_records)
df = score_df(df)
all_dfs.append(df)

# KPI cards (latest session)
latest = df.iloc[0]
st.subheader("🏷️ Key Metrics")
c1, c2, c3 = st.columns(3)
c1.metric("Duration (s)", f"{latest.total_duration:.1f}")
c2.metric("Speech Rate",   f"{latest.speech_rate:.2f} w/s")
c3.metric("Risk Score (%)",f"{latest.risk_score:.1f}%")

# Trend if multiple
if len(all_dfs) > 1:
    st.subheader("📈 Longitudinal Trend")
    plot_trend(all_dfs)

# Playback & transcript
st.subheader("🔊 Playback & Transcript")
st.audio(files[0].read())
st.write(latest.transcript)

# Interactive summary table
summary = df[[
    "filename","total_duration","speech_dur",
    "total_pause","num_pauses","speech_rate","pitch_mean","rms","risk_score"
]].copy()
summary.columns = [
    "File","Total (s)","Spoken (s)","Pause (s)","# Pauses",
    "Rate (w/s)","Pitch Mean","Loudness (RMS)","Risk (%)"
]
st.subheader("🔍 Extracted Features & Risk Scores")
gb = GridOptionsBuilder.from_dataframe(summary)
gb.configure_column("Risk (%)", cellStyle={
    "function":
    f"params.value >= {risk_thresh} ? {{'color':'red'}} : {{'color':'green'}}"
})
AgGrid(summary, gridOptions=gb.build(), enable_enterprise_modules=False)

# Pause histogram & interactive scatter
st.subheader("📊 Pause Distribution & Risk Mapping")
fig_hist, axh = plt.subplots()
y, sr         = librosa.load(df.tmp_path.iloc[0], sr=None, mono=True)
ints          = librosa.effects.split(y, top_db=25)
pause_lens    = [(ints[i][0]-ints[i-1][1])/sr for i in range(1,len(ints))]
axh.hist(pause_lens, bins=20, edgecolor="k", alpha=0.7)
axh.set_xlabel("Pause Length (s)")
axh.set_ylabel("Count")
st.pyplot(fig_hist)

fig_px = px.scatter(df, x="total_pause", y="risk_score",
                    color=df.risk_score>=risk_thresh,
                    color_discrete_map={True:"red",False:"green"},
                    labels={"color":"High Risk"})
fig_px.add_hline(y=risk_thresh, line_dash="dash")
st.plotly_chart(fig_px, use_container_width=True)

# Expanders for detail
st.subheader("📑 Detailed Analysis Report")
with st.expander("Project Overview"):
    st.write("...")

with st.expander("Methodology"):
    st.write("...")

with st.expander("Glossary"):
    st.write("...")

with st.expander("Results & Figures"):
    st.write("Histogram and interactive scatter above.")

with st.expander("Future Work"):
    st.write("...")

# Downloads
csv     = df.to_csv(index=False).encode("utf-8")
pdf_buf = make_pdf(df, fig_hist, fig_px, risk_thresh)

st.download_button("Download CSV", csv, "cognitive_report.csv", "text/csv")
st.download_button("Download PDF Report", pdf_buf, "cognitive_report.pdf", "application/pdf")
