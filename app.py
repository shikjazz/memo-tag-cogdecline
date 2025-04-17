import os
import tempfile
import warnings

import streamlit as st
import librosa
import numpy as np
import pandas as pd
import speech_recognition as sr
from pydub import AudioSegment

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Initialize recognizer
_recognizer = sr.Recognizer()

def transcribe(path):
    # Convert to WAV if needed
    ext = os.path.splitext(path)[1].lower()
    if ext != ".wav":
        wav_path = path.rsplit(".", 1)[0] + ".wav"
        AudioSegment.from_file(path).export(wav_path, format="wav")
        path = wav_path
    with sr.AudioFile(path) as src:
        audio = _recognizer.record(src)
    try:
        return _recognizer.recognize_google(audio).lower()
    except sr.UnknownValueError:
        return ""

# -----------------------------------------------------------------------------
# Feature extraction function
def extract_features(path):
    y, sr_rate = librosa.load(path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr_rate)
    intervals = librosa.effects.split(y, top_db=30)
    num_pauses = len(intervals) - 1
    total_pauses = duration - sum((e - s) / sr_rate for s, e in intervals)
    # Pitch variability via pyin
    f0, _, _ = librosa.pyin(y,
                             fmin=librosa.note_to_hz("C2"),
                             fmax=librosa.note_to_hz("C7"))
    pitch_var = float(np.nanstd(f0))

    transcript = transcribe(path)
    words = transcript.split()
    hesitations = sum(words.count(h) for h in ("um", "uh", "er", "ah"))
    speech_rate = len(words) / duration if duration > 0 else 0

    return {
        "filename": os.path.basename(path),
        "transcript": transcript,
        "num_pauses": num_pauses,
        "total_pause_duration": total_pauses,
        "speech_rate": speech_rate,
        "hesitation_count": hesitations,
        "pitch_variability": pitch_var,
    }

# -----------------------------------------------------------------------------
# Anomaly detection
def detect_anomalies(df, eps=1.0, contamination=0.2):
    feature_cols = [
        "num_pauses",
        "total_pause_duration",
        "speech_rate",
        "hesitation_count",
        "pitch_variability",
    ]
    X = df[feature_cols].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)

    db = DBSCAN(eps=eps, min_samples=2).fit(X_scaled)
    df["dbscan_outlier"] = db.labels_ == -1

    iso = IsolationForest(contamination=contamination, random_state=42).fit(X_scaled)
    df["iforest_anomaly"] = iso.predict(X_scaled) == -1

    return df

# -----------------------------------------------------------------------------
# Streamlit app layout
st.title("MemoTag Cognitive Decline Detection App")
st.markdown(
    "Upload one or multiple audio files (wav, mp3, m4a) to analyze for cognitive decline indicators."
)

uploaded_files = st.file_uploader(
    "Choose audio files", type=["wav", "mp3", "m4a"], accept_multiple_files=True
)

if uploaded_files:
    temp_dir = tempfile.mkdtemp()
    records = []
    for uploaded in uploaded_files:
        tmp_path = os.path.join(temp_dir, uploaded.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        records.append(extract_features(tmp_path))

    df = pd.DataFrame(records).set_index("filename")
    df = detect_anomalies(df)

    st.subheader("Feature Table with Anomaly Flags")
    st.dataframe(df)

    # Plot: Pause Duration Distribution
    st.subheader("Pause Duration Distribution")
    fig1, ax1 = plt.subplots()
    ax1.hist(df["total_pause_duration"], bins=5)
    ax1.set_xlabel("Pause Duration (s)")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)

    # Plot: Speech Rate per Clip
    st.subheader("Speech Rate per Clip")
    fig2, ax2 = plt.subplots()
    df["speech_rate"].plot.bar(ax=ax2)
    ax2.set_ylabel("Words/sec")
    st.pyplot(fig2)

    # Plot: Pitch Variability vs Pause Duration
    st.subheader("Pitch Variability vs Pause Duration")
    fig3, ax3 = plt.subplots()
    ax3.scatter(df["total_pause_duration"], df["pitch_variability"])
    ax3.set_xlabel("Pause Duration (s)")
    ax3.set_ylabel("Pitch Variability (Hz)")
    st.pyplot(fig3)

    # Executive Summary
    st.subheader("Executive Summary Report")
    st.markdown(
        """
**Most Insightful Features:**
- **Total Pause Duration:** Longer silences often indicate word-finding difficulty.
- **Pitch Variability:** Reduced or erratic pitch range may reflect cognitive stress.

**Modeling Approach:**
- **DBSCAN** for clustering speech profiles and flagging low-density outliers.
- **Isolation Forest** for anomaly scoring in high-dimensional feature space.

**Next Steps:**
1. Increase sample size (10+ clips) for robust clustering.
2. Implement semantic recall checks via embeddings.
3. Validate thresholds with a neurologist.
4. Deploy as an API endpoint for real-time risk scoring.
"""
    )
else:
    st.info("Please upload at least one audio file to begin analysis.")
