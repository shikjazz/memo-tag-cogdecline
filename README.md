# 📋 MemoTag Cognitive Decline Detection

![Streamlit App Screenshot](docs/screenshot.png)

## 🚀 Overview

**MemoTag** is a Streamlit‑based clinical research tool that uses **speech‑pause analysis** to flag potential early signs of cognitive decline. By combining OpenAI’s Whisper transcription, Librosa voice‑activity detection (VAD), and anomaly‑detection scoring (DBSCAN + IsolationForest), it computes a simple 0–100 “risk score.” It also generates a clinician‑ready, multi‑page PDF report.

---

## ⚙️ Features

- **Accurate ASR** via OpenAI Whisper  
- **Speech‑Pause Analysis**  
  - Total pause duration  
  - Number of pauses  
  - Pause‑length distribution histogram  
- **Prosodic Metric**: speech‑rate (words/sec)  
- **Anomaly‑based Scoring** (0–100%) with a configurable “high‑risk” threshold  
- **Interactive Dashboard**  
  - KPI cards (Duration, Speech Rate, Risk Score)  
  - Playback + transcript  
  - AgGrid table with conditional coloring  
  - Pause vs. Risk scatterplot  
  - Expandable Detailed Analysis (Overview, Methodology, Glossary, Future Work)  
- **Downloadable Reports**  
  - CSV of raw features  
  - **4‑page PDF**: Overview, Methodology, Results & Figures, Glossary & Next Steps  

---

## 🖥️ Demo

> **Live app:** [your‑streamlit‑cloud‑url]  
> ![App Screenshot](docs/screenshot.png)

---

## 📦 Installation

1. **Clone** this repository  
   ```bash
   git clone https://github.com/your‑org/memo‑tag‑cogdecline.git
   cd memo‑tag‑cogdecline
Create & activate a Python venv

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
(Optional) If you get FFmpeg errors, install it via your package manager:

bash
Copy
Edit
# macOS (Homebrew)
brew install ffmpeg
# Ubuntu/Debian
sudo apt-get install ffmpeg
▶️ Running Locally
bash
Copy
Edit
streamlit run app.py
Open http://localhost:8501 in your browser

Upload one or more audio files (.wav, .mp3, .m4a)

Explore the dashboard, tweak the “High‑risk threshold” slider, and download reports.

🔧 Configuration
Choose your Whisper model size (tiny, base, small, medium, large)

Select transcription language (e.g. en, hi, fr, es)

Adjust high‑risk threshold (0–100%) to tune sensitivity

All controls are in the sidebar.

📈 Example Workflow
Upload a spoken word list or story.

Watch the transcript appear.

See your total pause time and pause count.

View the pause‑length distribution histogram.

Check your risk score and see if it exceeds your threshold.

Download a 4‑page PDF report to share with clinicians.

🤝 Contributing
Fork the repo

Create a feature branch: git checkout -b feature/YourFeature

Commit your changes: git commit -m "feat: add X"

Push to your fork: git push origin feature/YourFeature

Open a Pull Request against main—include screenshots or sample audio.

📜 License
This project is released under the MIT License.

📞 Contact
Shikhar Sharma

GitHub: @shikjazz

Email: sg7039@srmist.edu.in

Built with ❤️ using Streamlit & OpenAI Whisper

yaml
Copy
Edit

---

### Next Steps

1. **Add the file**:
   ```bash
   git add README.md
   git commit -m "docs: add comprehensive README"
   git push origin main
(Optional) Create a docs/screenshot.png folder for your key screenshot, or update the path in the Markdown.
