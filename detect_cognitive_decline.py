# detect_cognitive_decline.py
import os, argparse
import librosa, numpy as np, pandas as pd, whisper
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load Whisper once
print("Loading Whisper model…")
_whisper_model = whisper.load_model("small")

def transcribe_audio_whisper(file_path):
    return _whisper_model.transcribe(file_path)["text"].lower()

def extract_features(file_path, transcript=None, top_db=30):
    y, sr_rate = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr_rate)
    intervals = librosa.effects.split(y, top_db=top_db)
    total_speech = sum((e-s) for s,e in intervals)/sr_rate
    total_pauses = max(duration - total_speech, 0)
    num_pauses = len(intervals)-1
    f0, _, _ = librosa.pyin(y,
                            fmin=librosa.note_to_hz("C2"),
                            fmax=librosa.note_to_hz("C7"))
    pitch_var = float(np.nanstd(f0))
    if transcript is None:
        transcript = transcribe_audio_whisper(file_path)
    words = transcript.split()
    num_words = len(words)
    hesitations = sum(words.count(h) for h in ("um","uh","er","ah"))
    speech_rate = num_words/duration if duration>0 else 0
    recall_issues = 0
    return {
        "duration": duration,
        "num_pauses": num_pauses,
        "total_pause_duration": total_pauses,
        "speech_rate": speech_rate,
        "hesitation_count": hesitations,
        "pitch_variability": pitch_var,
        "recall_issues": recall_issues
    }

def build_feature_dataframe(audio_dir):
    rows=[]
    for fn in os.listdir(audio_dir):
        if fn.lower().endswith((".wav",".mp3",".m4a")):
            path=os.path.join(audio_dir,fn)
            transcript=transcribe_audio_whisper(path)
            feats=extract_features(path,transcript=transcript)
            feats["filename"]=fn
            rows.append(feats)
    return pd.DataFrame(rows).set_index("filename")

def detect_anomalies(df, feature_cols):
    X = df[feature_cols].fillna(0).values
    Xs = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=1.2, min_samples=2)
    df["dbscan_label"] = db.fit_predict(Xs)
    df["dbscan_outlier"] = df["dbscan_label"] == -1
    iso = IsolationForest(contamination=0.2, random_state=42)
    df["iforest_score"] = iso.fit_predict(Xs)
    df["iforest_anomaly"] = df["iforest_score"] == -1
    return df

def plot_histogram(df, column, bins=20):
    plt.figure()
    plt.hist(df[column], bins=bins)
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

def plot_pca_scatter(df, feature_cols):
    pca = PCA(n_components=2)
    comps = pca.fit_transform(df[feature_cols].fillna(0))
    plt.figure()
    plt.scatter(comps[:,0], comps[:,1])
    plt.title("PCA Projection of Speech Features")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

def plot_anomaly_box(df):
    plt.figure()
    plt.boxplot(df["iforest_score"])
    plt.title("Isolation Forest Anomaly Scores")
    plt.show()

def main():
    parser=argparse.ArgumentParser(
        description="Detect cognitive‑decline risk from voice samples.")
    parser.add_argument("--audio-dir", required=True,
                        help="Folder of .wav/.mp3 samples")
    args=parser.parse_args()

    df = build_feature_dataframe(args.audio_dir)
    cols=["num_pauses","total_pause_duration","speech_rate",
          "hesitation_count","pitch_variability","recall_issues"]
    df = detect_anomalies(df, cols)

    print(df[cols+["dbscan_outlier","iforest_anomaly"]])
    plot_histogram(df, "total_pause_duration")
    plot_histogram(df, "hesitation_count")
    plot_pca_scatter(df, cols)
    plot_anomaly_box(df)

if __name__=="__main__":
    main()
