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

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="MemoTag Cognitive Decline", layout="wide")

# ─── Whisper loader ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_whisper(model_name="base"):
    return whisper.load_model(model_name)

# ─── Feature extraction ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def extract_features(audio_bytes: bytes, language: str, model_name: str):
    try:
        # write to temporary WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
            seg.export(tmp.name, format="wav")
            path = tmp.name

        # 1) ASR via Whisper
        model = load_whisper(model_name)
        res = model.transcribe(path, language=language)
        transcript = res["text"].strip()

        # 2) Pause detection
        y, sr = librosa.load(path, sr=None, mono=True)
        intervals = librosa.effects.split(y, top_db=25)
        speech_dur = sum((e - s) for s, e in intervals) / sr
        total_dur = len(y) / sr
        total_pause = total_dur - speech_dur
        num_pauses = max(len(intervals) - 1, 0)

        # 3) MFCC statistics
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = mfcc.mean(axis=1)
        mfcc_stds = mfcc.std(axis=1)

        # 4) Additional voice features
        zcr = librosa.feature.zero_crossing_rate(y)
        centroid = librosa.feature.spectral_centroid(y, sr=sr)

        feats = {
            "filename": tmp.name.split("/")[-1],
            "transcript": transcript,
            "total_duration": total_dur,
            "total_pause": total_pause,
            "num_pauses": num_pauses,
            "zcr_mean": float(zcr.mean()),
            "zcr_std": float(zcr.std()),
            "centroid_mean": float(centroid.mean()),
            "centroid_std": float(centroid.std()),
        }
        for i in range(13):
            feats[f"mfcc_{i+1}_mean"] = float(mfcc_means[i])
            feats[f"mfcc_{i+1}_std"]  = float(mfcc_stds[i])
        return feats

    except Exception as e:
        # if anything goes wrong, return NaNs
        return {
            "filename": None,
            "transcript": f"ERROR: {e}",
            "total_duration": np.nan,
            "total_pause": np.nan,
            "num_pauses": np.nan,
            "zcr_mean": np.nan,
            "zcr_std": np.nan,
            "centroid_mean": np.nan,
            "centroid_std": np.nan,
            **{f"mfcc_{i+1}_mean": np.nan for i in range(13)},
            **{f"mfcc_{i+1}_std":  np.nan for i in range(13)},
        }

# ─── Scoring ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def score_df(
    df: pd.DataFrame,
    eps: float, min_samples: int,
    contamination: float,
    w_pause: float, w_count: float, w_iso: float
) -> pd.DataFrame:
    X = df[["total_pause", "num_pauses"]].fillna(0)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    iso = IsolationForest(contamination=contamination).fit(X)

    df["dbscan_label"] = db.labels_
    df["iso_score"]    = iso.decision_function(X)

    # safe normalization
    pmax = df.total_pause.max() or 1
    cmax = df.num_pauses.max() or 1
    imax = df.iso_score.max()    or 1

    df["pause_norm"] = df.total_pause.apply(lambda v: v/pmax)
    df["count_norm"] = df.num_pauses.apply(lambda v: v/cmax)
    df["iso_norm"]   = df.iso_score.apply(lambda v: 1 - (v/imax))

    # composite risk
    df["risk_score"] = (
        df.pause_norm * w_pause +
        df.count_norm * w_count +
        df.iso_norm   * w_iso
    ) * 100
    df["risk_score"] = df.risk_score.clip(0,100).round(1)
    return df

# ─── PDF report builder ─────────────────────────────────────────────────────────
def make_pdf(df: pd.DataFrame, fig) -> io.BytesIO:
    buf = io.BytesIO()
    c   = canvas.Canvas(buf, pagesize=letter)
    w,h = letter

    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, h - 30, "MemoTag Cognitive Decline Report")

    text = c.beginText(30, h - 60)
    text.setFont("Helvetica", 10)
    row = df.iloc[0]
    for k in ["filename","total_pause","num_pauses","risk_score"]:
        text.textLine(f"{k}: {row[k]}")
    c.drawText(text)

    # embed the scatter plot
    img = io.BytesIO()
    fig.savefig(img, format="PNG", bbox_inches="tight")
    img.seek(0)
    c.drawImage(ImageReader(img), 30, h - 300, width=550, preserveAspectRatio=True)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf

# ─── UI ────────────────────────────────────────────────────────────────────────
st.title("📋 MemoTag Cognitive Decline Detection")

# — Sidebar controls —
st.sidebar.header("Settings")
language      = st.sidebar.selectbox("Transcription Language", ["en","hi","fr","es"], index=0)
model_name    = st.sidebar.selectbox("Whisper Model",        ["tiny","base","small","medium","large"], index=1)
eps           = st.sidebar.slider("DBSCAN ε",               0.1, 2.0, 0.5, step=0.1)
min_samples   = st.sidebar.slider("DBSCAN min_samples",     1, 10,   2,   step=1)
contamination = st.sidebar.slider("IsoForest cont.",        0.01,0.5,  0.1, step=0.01)
w_pause       = st.sidebar.slider("Weight: pause",          0.0, 1.0,  0.5, step=0.05)
w_count       = st.sidebar.slider("Weight: pauses",         0.0, 1.0,  0.3, step=0.05)
w_iso         = st.sidebar.slider("Weight: anomaly",        0.0, 1.0,  0.2, step=0.05)

if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()

# — File uploader —
files = st.file_uploader(
    "Upload audio (wav/mp3/m4a)",
    type=["wav","mp3","m4a"],
    accept_multiple_files=True
)
if not files:
    st.info("🚀 Please upload at least one audio file to begin.")
    st.stop()

# read bytes
audio_bytes_list = [f.read() for f in files]

# extract & score
records = []
for audio_bytes in audio_bytes_list:
    with st.spinner("🔍 Extracting features…"):
        feats = extract_features(audio_bytes, language, model_name)
    records.append(feats)

df = pd.DataFrame(records)
df = score_df(df, eps, min_samples, contamination, w_pause, w_count, w_iso)

# — Audio playback & transcript —
st.subheader("🔊 Audio Playback & Transcript")
st.audio(audio_bytes_list[0])  # click‑able player
st.markdown("**Transcript:**")
st.write(df.transcript.iloc[0])

# — Features table —
st.subheader("🔍 Extracted Features & Scores")
# show full table (multiple rows & columns)
st.dataframe(df, use_container_width=True)

# — Scatter plot —
st.subheader("🗺️ Pause vs. Risk Score")
fig, ax = plt.subplots()
ax.scatter(df.total_pause, df.risk_score, s=60, edgecolor="k", alpha=0.7)
ax.set_xlabel("Total Pause Duration (s)")
ax.set_ylabel("Risk Score (0–100)")
st.pyplot(fig)

# — Downloads & PDF preview —
st.subheader("⬇️ Downloads & PDF Preview")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "cognitive_report.csv", "text/csv")

pdf_buf = make_pdf(df, fig)
st.download_button("Download PDF Report", pdf_buf, "cognitive_report.pdf", "application/pdf")

# embed the PDF inline
b64 = base64.b64encode(pdf_buf.getvalue()).decode()
pdf_html = f'<iframe src="data:application/pdf;base64,{b64}" width="700" height="500" type="application/pdf"></iframe>'
st.markdown(pdf_html, unsafe_allow_html=True)

# — Executive summary —
st.subheader("📝 Executive Summary")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"- **Total Pause:**  {df.total_pause.iloc[0]:.1f}s")
    st.markdown(f"- **Num Pauses:**  {int(df.num_pauses.iloc[0])}")
with c2:
    st.markdown("**Cognitive Risk Score**")
    st.markdown(
        f"<h1 style='color:#d6336c'>{df.risk_score.iloc[0]}</h1>",
        unsafe_allow_html=True
    )

st.markdown(
    """
**Next Steps**  
1. Validate thresholds with clinicians  
2. Deploy as a REST API endpoint  
3. Integrate longitudinal tracking dashboard  
4. Optimize Whisper model for CPU/GPU  
"""
)
