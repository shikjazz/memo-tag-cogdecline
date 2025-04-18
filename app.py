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

import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MemoTag Cognitive Decline", layout="wide")

# ── Load Whisper model once ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_whisper(model_name="base"):
    return whisper.load_model(model_name)

# ── Feature extraction (no parselmouth) ───────────────────────────────────────
@st.cache_data(show_spinner=False)
def extract_features(audio_bytes: bytes, language: str, model_name: str):
    # write to temp .wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        seg.export(tmp.name, format="wav")
        path = tmp.name

    # whisper ASR
    model = load_whisper(model_name)
    res = model.transcribe(path, language=language)
    transcript = res["text"].strip()

    # pause detection
    y, sr = librosa.load(path, sr=None, mono=True)
    intervals = librosa.effects.split(y, top_db=25)
    speech_dur = sum((e - s) for s, e in intervals) / sr
    total_dur = len(y) / sr
    total_pause = total_dur - speech_dur
    num_pauses = max(len(intervals) - 1, 0)

    return {
        "filename": tmp.name.split("/")[-1],
        "transcript": transcript,
        "total_duration": total_dur,
        "total_pause": total_pause,
        "num_pauses": num_pauses,
    }

# ── Normative risk scoring ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def score_df(df: pd.DataFrame) -> pd.DataFrame:
    # define “max acceptable” norms
    MAX_PAUSE_SEC = 30.0    # e.g. 30s
    MAX_PAUSE_COUNT = 50.0  # e.g. 50 pauses

    def compute_risk(row):
        p_score = min(1.0, row.total_pause / MAX_PAUSE_SEC)
        c_score = min(1.0, row.num_pauses / MAX_PAUSE_COUNT)
        return round((p_score + c_score) / 2 * 100, 1)

    df["risk_score"] = df.apply(compute_risk, axis=1)
    return df

# ── PDF builder ───────────────────────────────────────────────────────────────
def make_pdf(df: pd.DataFrame, fig) -> io.BytesIO:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(30, h - 30, "MemoTag Cognitive Decline Report")

    text = c.beginText(30, h - 70)
    text.setFont("Helvetica", 11)
    row = df.iloc[0]
    text.textLine(f"File:        {row.filename}")
    text.textLine(f"Duration:    {row.total_duration:.1f}s")
    text.textLine(f"Total Pause: {row.total_pause:.1f}s")
    text.textLine(f"# Pauses:    {int(row.num_pauses)}")
    text.textLine(f"Risk Score:  {row.risk_score}%")
    c.drawText(text)

    # embed the scatter chart
    img = io.BytesIO()
    fig.savefig(img, format="PNG", bbox_inches="tight")
    img.seek(0)
    c.drawImage(ImageReader(img), 30, h - 350, width=550, preserveAspectRatio=True)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📋 MemoTag Cognitive Decline Detection")

# Sidebar
st.sidebar.header("Settings")
language   = st.sidebar.selectbox("Transcription Language", ["en","hi","fr","es"], index=0)
model_name = st.sidebar.selectbox("Whisper Model",        ["tiny","base","small","medium","large"], index=1)
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()

# File upload
files = st.file_uploader("Upload audio (wav/mp3/m4a)", type=["wav","mp3","m4a"], accept_multiple_files=True)
if not files:
    st.info("Upload at least one audio file to begin.")
    st.stop()

# Read & process
audio_bytes_list = [f.read() for f in files]
records = []
for b in audio_bytes_list:
    with st.spinner("Extracting features…"):
        records.append(extract_features(b, language, model_name))

df = pd.DataFrame(records)
df = score_df(df)

# ── Playback & transcript ────────────────────────────────────────────────────
st.subheader("🔊 Playback & Transcript")
st.audio(audio_bytes_list[0])  # auto MIME detection
st.markdown(f"**Transcript:** {df.transcript.iloc[0]}")

# ── Features table ──────────────────────────────────────────────────────────
st.subheader("🔍 Extracted Features & Risk Scores")
summary = df[[
    "filename",
    "total_duration",
    "total_pause",
    "num_pauses",
    "risk_score"
]].copy()
summary.columns = ["File", "Duration (s)", "Pause (s)", "# Pauses", "Risk (%)"]
st.table(summary)

# ── Scatter plot ────────────────────────────────────────────────────────────
st.subheader("🗺️ Pause vs. Risk")
fig, ax = plt.subplots()
ax.scatter(df.total_pause, df.risk_score, s=80, edgecolor="k", alpha=0.7)
ax.set_xlabel("Total Pause (s)")
ax.set_ylabel("Risk (%)")
ax.set_xlim(left=0)
ax.set_ylim(0, 100)
st.pyplot(fig)

# ── Downloads & PDF preview ─────────────────────────────────────────────────
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "cognitive_report.csv", "text/csv")

pdf_buf = make_pdf(df, fig)
st.download_button("Download PDF Report", pdf_buf, "cognitive_report.pdf", "application/pdf")

# embed inline PDF
b64 = base64.b64encode(pdf_buf.getvalue()).decode("utf-8")
pdf_display = f"""
<iframe src="data:application/pdf;base64,{b64}" width="100%" height="600px" type="application/pdf"></iframe>
"""
st.markdown(pdf_display, unsafe_allow_html=True)

# ── Executive summary ────────────────────────────────────────────────────────
st.subheader("📝 Executive Summary")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"- **Total Pause:** {df.total_pause.iloc[0]:.1f}s")
    st.markdown(f"- **# Pauses:** {int(df.num_pauses.iloc[0])}")
with c2:
    st.markdown("**Cognitive Risk Score**")
    st.markdown(
        f"<h1 style='color:#d6336c'>{df.risk_score.iloc[0]}%</h1>",
        unsafe_allow_html=True
    )

st.markdown("""
**Next Steps**  
1. Validate thresholds with clinicians  
2. Deploy as a REST API endpoint  
3. Integrate longitudinal tracking dashboard  
4. Optimize Whisper model for CPU/GPU  
""")
