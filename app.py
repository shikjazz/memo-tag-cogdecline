import streamlit as st
st.set_option("server.fileWatcherType", "none")
import os, tempfile, warnings
import librosa, numpy as np, pandas as pd, whisper
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")

model = load_whisper_model()

def transcribe(path):
    return model.transcribe(path)["text"].lower()

def extract_features(path):
    y, sr = librosa.load(path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    intervals = librosa.effects.split(y, top_db=30)
    num_pauses = len(intervals) - 1
    total_pauses = duration - sum((e - s) / sr for s, e in intervals)
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

def detect_anomalies(df, eps=1.0, contamination=0.2):
    features = ["num_pauses", "total_pause_duration", "speech_rate",
                "hesitation_count", "pitch_variability"]
    X = StandardScaler().fit_transform(df[features].fillna(0))
    df["dbscan_outlier"] = DBSCAN(eps=eps, min_samples=2).fit_predict(X) == -1
    df["iforest_anomaly"] = IsolationForest(
        contamination=contamination, random_state=42
    ).fit_predict(X) == -1
    return df

st.title("MemoTag Cognitive Decline Detection App")
st.markdown("Upload audio files to analyze for cognitive decline indicators.")

uploaded = st.file_uploader(
    "Choose audio files", type=["wav","mp3","m4a"], accept_multiple_files=True
)

if uploaded:
    tmp = tempfile.mkdtemp()
    records = []
    for u in uploaded:
        p = os.path.join(tmp, u.name)
        with open(p, "wb") as f: f.write(u.getbuffer())
        records.append(extract_features(p))
    df = pd.DataFrame(records).set_index("filename")
    df = detect_anomalies(df)
    st.subheader("Features & Anomalies")
    st.dataframe(df)
    # plots...
    # executive summary...
else:
    st.info("Please upload at least one audio file.")
