"""PULSE PILOT AI - core logic (symptom triage rules, ML benchmark, AI chat, PDF report)."""
import os
import io
import re
from datetime import datetime

import numpy as np
import pandas as pd

APP_NAME = "PULSE PILOT AI"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"

SYSTEM_PROMPT = """You are PULSE PILOT AI, a cautious health-education assistant. Provide general information only.
Never diagnose or prescribe medication/doses. For severe difficulty breathing, chest pain, fainting, confusion,
stroke-like symptoms, or a severe allergic reaction, advise contacting local emergency services immediately.
Encourage personalized advice from a qualified clinician. Do not ask for identifying information.
Be clear, concise and cautious. If asked about anything unrelated to health education, politely decline."""

# Matched as lowercase substrings against normalised symptom names, so it works for both the
# built-in fallback list ("Chest pain") and dataset columns ("chest_pain", "breathlessness", ...).
URGENT_KEYWORDS = (
    "chest pain", "breathless", "difficulty breathing", "shortness of breath",
    "faint", "confusion", "altered sensorium", "allergic", "anaphylaxis", "coma",
    "slurred speech", "one body side", "paralysis", "loss of consciousness",
    "blood in sputum", "stomach bleeding",
)

# Free-text patterns used by the AI chat to show a safety banner.
EMERGENCY_PATTERNS = re.compile(
    r"chest pain|can'?t breathe|cannot breathe|difficulty breathing|short(ness)? of breath|"
    r"faint(ed|ing)|unconscious|stroke|seizure|overdose|anaphyla|severe (bleeding|allergic)|"
    r"suicid|kill myself|self[- ]harm",
    re.IGNORECASE,
)

MAX_PROMPT_CHARS = 1000


def _norm(text):
    return re.sub(r"[_\s]+", " ", str(text)).strip().lower()


def is_urgent_symptom(symptom):
    s = _norm(symptom)
    return any(k in s for k in URGENT_KEYWORDS)


def looks_like_emergency(text):
    return bool(EMERGENCY_PATTERNS.search(text or ""))


def assess_symptoms(symptoms, duration, severity, worsening, age_group):
    chosen = set(symptoms or [])
    urgent = any(is_urgent_symptom(s) for s in chosen)
    same_day = severity == "Severe" or worsening == "Yes" or (age_group in {"Under 18", "65+"} and severity == "Moderate")
    if urgent:
        heading = "Some selected symptoms can be urgent"
        summary = "One or more selected warning signs may need emergency assessment. This app cannot judge how serious your situation is."
        next_step = "Seek emergency medical help now if symptoms are happening now, are severe, or feel unusual for you."
    elif same_day:
        heading = "Consider prompt medical advice"
        summary = "You marked symptoms as severe or worsening, or selected an age group that may need extra caution."
        next_step = "Contact a qualified healthcare professional or local urgent-care service today. If symptoms rapidly worsen, seek emergency help."
    elif not chosen:
        heading = "Start with what you notice"
        summary = "No symptoms were selected, so there is not enough information to prepare a check-in summary."
        next_step = "Select symptoms or contact a healthcare professional if you are concerned."
    else:
        heading = "Keep track and use your judgement"
        summary = f"You reported {len(chosen)} symptom type(s), duration: {duration.lower()}, severity: {severity.lower()}."
        next_step = "Monitor changes and contact a clinician if symptoms persist, worsen, or concern you."
    return {"urgent": urgent, "same_day": same_day, "heading": heading, "summary": summary, "next_step": next_step}


def emergency_warning():
    return "For severe difficulty breathing, chest pain, fainting, confusion, stroke-like symptoms, or severe allergic reaction, contact local emergency services."


def get_health_topics():
    return [
        {"title": "Preparing for a medical appointment", "text": "Write down when symptoms began, what makes them better or worse, relevant history, allergies, and medicines or supplements. Bring questions you want answered.", "note": "A clinician can help decide which details matter."},
        {"title": "Fever: general information", "text": "Fever can occur for many reasons. Pay attention to how you feel overall, how long it lasts, hydration, and any other symptoms.", "note": "Seek urgent care for severe symptoms, breathing difficulty, confusion, stiff neck, or other alarming changes. Extra caution is needed for infants."},
        {"title": "Hydration basics", "text": "Fluids help replace water lost through activity, heat, fever, vomiting, or diarrhea. Small, frequent sips may be easier if you feel nauseated.", "note": "People with fluid restrictions should follow their clinician's advice."},
        {"title": "Understanding antibiotics", "text": "Antibiotics treat certain bacterial infections, not viral illnesses such as common colds. Treatment depends on clinical assessment.", "note": "Do not take leftover antibiotics or someone else's prescription."},
        {"title": "When to seek emergency help", "text": "Severe difficulty breathing, chest pain, fainting, severe confusion, stroke-like symptoms, or a severe allergic reaction may be emergencies.", "note": "Contact your local emergency number. Do not wait for an app response."},
        {"title": "Making sense of online health information", "text": "Look for author qualifications, date, cited evidence, and whether uncertainty is explained. Be wary of guaranteed cures and dramatic claims.", "note": "Online information cannot replace personalized medical assessment."},
    ]


class ModelWrapper:
    def __init__(self, model, features):
        self.model = model
        self.features = features

    def predict_candidates(self, symptoms, limit=3):
        chosen = set(symptoms)
        vector = pd.DataFrame([[1 if f in chosen else 0 for f in self.features]], columns=self.features)
        probs = self.model.predict_proba(vector)[0]
        labels = self.model.classes_
        idx = np.argsort(probs)[::-1][:limit]
        return [{"label": str(labels[i]), "score": float(probs[i])} for i in idx]


def load_ml_system():
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, "Data", "Training.csv")
    using_sample = False
    if not os.path.exists(path):
        path = os.path.join(root, "Data", "sample_training.csv")
        using_sample = True
    if not os.path.exists(path):
        raise FileNotFoundError("No dataset found. Add Data/Training.csv (or keep Data/sample_training.csv).")

    df = pd.read_csv(path).dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    if "prognosis" not in df.columns:
        raise ValueError("Training.csv must include a 'prognosis' target column.")

    # Many public symptom datasets contain heavily duplicated rows. Duplicates that land in both the
    # train and test split leak information and inflate accuracy, so remove them BEFORE splitting.
    rows_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=["prognosis"]).apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["prognosis"].astype(str).str.strip()
    if X.empty or y.nunique() < 2:
        raise ValueError("Training.csv needs feature columns and at least two target classes.")

    counts = y.value_counts()
    n_test = int(np.ceil(len(y) * 0.25))
    can_stratify = counts.min() >= 2 and n_test >= y.nunique()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if can_stratify else None
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=160, random_state=42, class_weight="balanced", n_jobs=-1),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }
    labels = sorted(y.unique())
    n_splits = int(min(5, counts.min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if n_splits >= 2 else None

    metrics, conf, fitted = {}, {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        entry = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
            "cv_mean": None, "cv_std": None,
        }
        if cv is not None:
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=1)
                entry["cv_mean"], entry["cv_std"] = float(scores.mean()), float(scores.std())
            except Exception:
                pass
        metrics[name] = entry
        fitted[name] = model
        conf[name] = confusion_matrix(y_test, pred, labels=labels)

    rf = fitted["Random Forest"]
    fi = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_}) \
        .sort_values("Importance", ascending=False).reset_index(drop=True)
    return {
        "models": {name: ModelWrapper(m, list(X.columns)) for name, m in fitted.items()},
        "metrics": metrics, "confusion": conf, "class_labels": labels,
        "symptoms": list(X.columns), "diseases": labels,
        "rows": len(df), "rows_raw": rows_raw, "duplicates_removed": rows_raw - len(df),
        "feature_importance": fi, "holdout_rows": len(X_test), "using_sample": using_sample, "cv_folds": n_splits if cv else 0,
    }


def _get_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            key = ""
    return key


def ask_ai(prompt, history=None):
    """Send the latest question (plus up to 6 recent turns) to OpenAI. `history` = [{"role","content"}]."""
    key = _get_api_key()
    if not key:
        return "**Live AI is not configured.** Add `OPENAI_API_KEY` as described in README.md. The other app features work without it."
    prompt = (prompt or "").strip()[:MAX_PROMPT_CHARS]
    if not prompt:
        return "Please type a question."
    messages = [{"role": m["role"], "content": m["content"]} for m in (history or [])[-6:]
                if m.get("role") in {"user", "assistant"}]
    messages.append({"role": "user", "content": prompt})
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        response = client.responses.create(model=model, instructions=SYSTEM_PROMPT, input=messages)
        return getattr(response, "output_text", "").strip() or "I couldn't create a response just now. Please try again."
    except Exception as exc:
        return "The AI service could not be reached. Check the API key, internet connection, and account access. Details: " + str(exc)[:180]


def build_pdf_report(report):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from xml.sax.saxutils import escape

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=44, leftMargin=44, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BrandTitle", parent=styles["Title"], textColor=colors.HexColor("#0e7490"), fontSize=22, leading=26, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="SmallNote", parent=styles["BodyText"], textColor=colors.HexColor("#475569"), fontSize=8.5, leading=12))
    cell = ParagraphStyle(name="Cell", parent=styles["BodyText"], fontSize=9.5, leading=13)

    story = [Paragraph(f"{APP_NAME} | Symptom Summary", styles["BrandTitle"]), Spacer(1, 6)]
    story.append(Paragraph(f"Generated {datetime.now():%d %b %Y, %H:%M}", styles["SmallNote"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Educational summary only. This document is not a diagnosis or medical advice.", styles["SmallNote"]))
    story.append(Spacer(1, 16))

    result = report.get("result", {})
    rows = [
        ("Symptoms", ", ".join(report.get("symptoms", [])) or "None selected"),
        ("Duration", report.get("duration", "Not specified")),
        ("Self-reported severity", report.get("severity", "Not specified")),
        ("Getting worse?", report.get("worsening", "Not specified")),
        ("Check-in", result.get("heading", "Not available")),
        ("Summary", result.get("summary", "Not available")),
        ("Suggested next step", result.get("next_step", "Not available")),
    ]
    data = [["Field", "Summary"]] + [[Paragraph(escape(a), cell), Paragraph(escape(str(b)), cell)] for a, b in rows]
    table = Table(data, colWidths=[125, 360], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0e7490")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#ecfeff")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([table, Spacer(1, 14)])

    preds = report.get("predictions", [])
    if preds:
        story.append(Paragraph("Dataset model candidates (not clinical probabilities)", styles["Heading2"]))
        for p in preds:
            story.append(Paragraph(f"{escape(p['label'])}: {p['score']:.1%} model score", styles["BodyText"]))
    story.extend([Spacer(1, 14), Paragraph("If symptoms are severe, worsening, persistent, or concerning, contact a qualified healthcare professional. For emergencies, contact local emergency services.", styles["SmallNote"])])
    doc.build(story)
    return buffer.getvalue()
