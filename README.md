# 🫀 PULSE PILOT AI

An AI-powered health-education web app built with Python and Streamlit. It combines a safety-first symptom navigator, a transparent machine-learning lab, an optional generative-AI health companion, and downloadable PDF summaries.

> **Live demo:** _add your Streamlit link here after deployment_

> ⚠️ **Educational project only.** PULSE PILOT AI is not a medical device and does not provide diagnosis or treatment. For emergencies, contact your local emergency services.

## Features
- **Symptom Navigator** – select symptoms, duration, severity and age group; get rule-based safety guidance (urgent warning signs are detected reliably across dataset and fallback symptom names).
- **Model Lab** – Decision Tree vs Random Forest vs SVM with accuracy, weighted precision/recall/F1, 5-fold cross-validation, feature importance and confusion matrices.
- **Leakage-safe evaluation** – duplicate rows are removed *before* the train/test split, so scores are not inflated.
- **AI Health Chat** (optional) – OpenAI-powered, multi-turn, with an emergency-keyword safety banner and a per-session question limit.
- **Health Library** – searchable plain-language health topics.
- **PDF report** – downloadable symptom summary.

## Tech stack
Python · Streamlit · scikit-learn · pandas · NumPy · Plotly · ReportLab · OpenAI API

## Project structure
```text
pulse-pilot-ai/
├── app.py                 # Streamlit UI
├── health_engine.py       # triage rules, ML pipeline, AI chat, PDF builder
├── requirements.txt
├── Data/sample_training.csv  # small synthetic demo dataset (included)
├── Data/Training.csv         # optional: your own dataset (git-ignored)
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

## Run locally
```bash
git clone https://github.com/<your-username>/pulse-pilot-ai.git
cd pulse-pilot-ai
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Open http://localhost:8501.

## Enable the AI chat (optional)
Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your key (this file is git-ignored), or set the `OPENAI_API_KEY` environment variable. Optionally set `OPENAI_MODEL` to change the model. Never commit your key.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io → **Create app** → pick the repo, branch `main`, main file `app.py`.
3. In **Advanced settings → Secrets**, paste `OPENAI_API_KEY = "sk-..."` (optional).
4. Click **Deploy**.

## Limitations
- The repository ships a small **synthetic demo dataset** (9 common conditions) so the app works out of the box. It is not real patient data. You can place your own `Data/Training.csv` (needs a `prognosis` column plus 0/1 symptom columns) and the app will use it automatically.
- Scores on any educational dataset do **not** reflect real-world clinical performance.
- Model "scores" are not calibrated probabilities.
- AI answers can be wrong and must never be used for emergencies.
