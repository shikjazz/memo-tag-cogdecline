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

@st.cache_resource(show_spinner=False)
def load_whisper(model_name="base"):
    return whisper.load_model(model_name)

@st.cache_data(show_spinner=False)
def extract_features(audio_bytes: bytes, language: str, model_name: str):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        seg.export(tmp.name, format="wav")
        path = tmp.name

    model = load_whisper(model_name)
    res = model.transcribe(path, language=language)
    transcript = res["text"].strip()

    y, sr = librosa.load(path, sr=None, mono=True)
    intervals = librosa.effects.split(y, top_db=25)
    speech_dur = sum((e - s) for s, e in intervals) / sr
    total_dur = len(y) / sr
    total_pause = total_dur - speech_dur
    num_pauses = max(len(intervals) - 1, 0)

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
    for i in range(13):
        feats[f"mfcc_{i+1}_mean"] = float(mfcc_means[i])
        feats[f"mfcc_{i+1}_std"]  = float(mfcc_stds[i])
    return feats

@st.cache_data(show_spinner=False)
def score_df(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 1:
        df["risk_score"] = ((df.total_pause / df.total_duration) * 100).round(1)
        return df

    X = df[["total_pause", "num_pauses"]]
    DBSCAN(eps=0.5, min_samples=2).fit(X)
    iso = IsolationForest(contamination=0.1).fit(X)
    df["iso_score"] = iso.decision_function(X)

    pmax, nmax, imax = df.total_pause.max(), df.num_pauses.max(), df.iso_score.max()
    df["pause_norm"] = df.total_pause.apply(lambda v: v/pmax if pmax>0 else 0)
    df["count_norm"] = df.num_pauses.apply(lambda v: v/nmax if nmax>0 else 0)
    df["iso_norm"]   = df.iso_score.apply(lambda v: 1 - (v/imax) if imax>0 else 0).fillna(0)

    w1, w2, w3 = 0.5, 0.3, 0.2
    df["risk_score"] = (
        df.pause_norm * w1 +
        df.count_norm * w2 +
        df.iso_norm   * w3
    ) * 100
    df["risk_score"] = df.risk_score.clip(0,100).round(1)
    return df

def make_pdf(df: pd.DataFrame, fig) -> io.BytesIO:
    buf = io.BytesIO()
    c   = canvas.Canvas(buf, pagesize=letter)
    w,h = letter

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, h-30, "MemoTag Cognitive Decline Report")

    # Transcript
    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, h-60, "Transcript:")
    text = c.beginText(30, h-80)
    text.setFont("Helvetica", 10)
    for line in df.transcript.iloc[0].split("\n"):
        text.textLine(line)
    c.drawText(text)

    # Summary table
    summary = df[["filename","total_duration","total_pause","num_pauses","risk_score"]].copy()
    summary.columns = ["File","Duration (s)","Pause (s)","Pauses Count","Risk (%)"]
    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, h-150, "Summary Table:")
    tbl = c.beginText(30, h-170)
    tbl.setFont("Helvetica", 10)
    tbl.textLine(" | ".join(summary.columns))
    tbl.textLine("-" * 80)
    for _, row in summary.iterrows():
        tbl.textLine(" | ".join(str(row[col]) for col in summary.columns))
    c.drawText(tbl)

    # Chart
    img = io.BytesIO()
    fig.savefig(img, format="PNG", bbox_inches="tight")
    img.seek(0)
    c.drawImage(ImageReader(img), 30, h-350, width=550, preserveAspectRatio=True)

    # Executive summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, h-500, "Executive Summary:")
    summ = c.beginText(30, h-520)
    summ.setFont("Helvetica", 10)
    summ.textLine(f"Total Pause:  {df.total_pause.iloc[0]:.1f} s")
    summ.textLine(f"Pauses Count: {int(df.num_pauses.iloc[0])}")
    summ.textLine(f"Risk Score:   {df.risk_score.iloc[0]:.1f} %")
    c.drawText(summ)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf

# --- Streamlit UI ---
st.title("📋 MemoTag Cognitive Decline Detection")

st.sidebar.header("Settings")
language   = st.sidebar.selectbox("Transcription Language", ["en","hi","fr","es"])
model_name = st.sidebar.selectbox("Whisper Model", ["tiny","base","small","medium","large"])
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()

files = st.file_uploader(
    "Upload audio (wav/mp3/m4a)",
    type=["wav","mp3","m4a"],
    accept_multiple_files=True
)
if not files:
    st.info("Upload at least one audio file to begin.")
    st.stop()

audio_bytes_list = [f.read() for f in files]
records = [extract_features(b, language, model_name) for b in audio_bytes_list]
df = score_df(pd.DataFrame(records))

st.subheader("🔊 Playback & Transcript")
st.audio(audio_bytes_list[0])
st.markdown(f"**Transcript:** {df.transcript.iloc[0]}")

summary = df[["filename","total_duration","total_pause","num_pauses","risk_score"]].copy()
summary.columns = ["File","Duration (s)","Pause (s)","Pauses Count","Risk (%)"]
st.subheader("🔍 Extracted Features & Risk Scores")
st.dataframe(summary, use_container_width=True)

st.subheader("🗺️ Pause vs. Risk Score")
fig, ax = plt.subplots()
ax.scatter(df.total_pause, df.risk_score, s=60, edgecolor="k", alpha=0.7)
ax.set_xlabel("Total Pause Duration (s)")
ax.set_ylabel("Risk Score (0–100)")
st.pyplot(fig)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "cognitive_report.csv", "text/csv")

pdf_buf = make_pdf(df, fig)
st.download_button("Download PDF Report", pdf_buf, "cognitive_report.pdf", "application/pdf")

st.subheader("📝 Executive Summary")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"- **Total Pause:** {df.total_pause.iloc[0]:.1f}s")
    st.markdown(f"- **Pauses Count:** {int(df.num_pauses.iloc[0])}")
with c2:
    st.markdown("**Cognitive Risk Score**")
    st.markdown(f"<h1 style='color:#d6336c'>{df.risk_score.iloc[0]:.1f}%</h1>", unsafe_allow_html=True)

st.markdown("""
**Next Steps**  
1. Validate thresholds with clinicians  
2. Deploy as a REST API endpoint  
3. Integrate longitudinal tracking dashboard  
4. Optimize Whisper model for CPU/GPU  
""")
