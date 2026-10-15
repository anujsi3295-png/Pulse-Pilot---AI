import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from health_engine import (
    load_ml_system, assess_symptoms, get_health_topics, ask_ai,
    build_pdf_report, emergency_warning, looks_like_emergency, APP_NAME
)

MAX_CHAT_TURNS = 10  # per browser session; protects your OpenAI bill on a public app

st.set_page_config(page_title="PULSE PILOT AI", page_icon="🫀", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#080d18;--panel:#101a2b;--line:rgba(148,163,184,.16);--muted:#94a3b8;--cyan:#67e8f9;--mint:#6ee7b7;--violet:#a5b4fc}
html,body,[class*="css"],.stApp{font-family:'DM Sans',sans-serif;background:var(--bg);color:#e5eefb}
[data-testid="stAppViewContainer"]{background:radial-gradient(ellipse at 12% 0%,rgba(8,145,178,.16),transparent 34%),radial-gradient(ellipse at 90% 10%,rgba(79,70,229,.14),transparent 30%),var(--bg)}
[data-testid="stSidebar"]{background:#0a1220;border-right:1px solid var(--line)}
.block-container{padding-top:1.8rem;max-width:1400px}
h1,h2,h3{font-family:'Manrope',sans-serif!important;letter-spacing:-.035em;color:#f1f5f9!important}
.hero{padding:30px 32px;border:1px solid rgba(103,232,249,.22);border-radius:24px;background:linear-gradient(120deg,rgba(14,116,144,.25),rgba(30,41,59,.72) 55%,rgba(67,56,202,.18));margin-bottom:24px}
.eyebrow{color:var(--cyan);font-size:.75rem;text-transform:uppercase;letter-spacing:.18em;font-weight:800}
.hero h1{font-size:clamp(2.4rem,4vw,4rem);line-height:1.03;margin:.55rem 0 .7rem}
.hero p{color:#b6c5d9;max-width:760px;font-size:1rem;line-height:1.75}
.panel{border:1px solid var(--line);border-radius:18px;padding:20px;background:rgba(16,26,43,.78)}
.metric{font-family:'Manrope',sans-serif;font-size:1.7rem;font-weight:800;color:var(--cyan)}
.muted{color:var(--muted);font-size:.92rem}
.stButton>button,.stDownloadButton>button{border-radius:11px;border:1px solid rgba(103,232,249,.25);background:linear-gradient(110deg,#0e7490,#155e75);color:white;font-weight:700;padding:.65rem 1rem}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--cyan);color:white}
[data-testid="stMetric"]{background:rgba(16,26,43,.78);border:1px solid var(--line);padding:16px;border-radius:16px}
[data-testid="stMetricValue"]{color:var(--cyan)}
hr{border-color:var(--line)}
div[data-testid="stTabs"] button{font-weight:700}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Preparing and evaluating machine-learning models...")
def get_system():
    try:
        return load_ml_system()
    except Exception as exc:
        return {"error": str(exc)}

system = get_system()

with st.sidebar:
    st.markdown("# 🫀 PULSE PILOT AI")
    st.caption("AI HEALTH COMPANION")
    st.markdown("---")
    page = st.radio("NAVIGATION", [
        "Overview", "Symptom Navigator", "Model Lab", "AI Health Chat",
        "Health Library", "PDF Report", "About"
    ])
    st.markdown("---")
    if isinstance(system, dict) and "error" not in system:
        st.success("Dataset loaded")
        st.caption(f"{len(system['symptoms'])} symptom features · {len(system['diseases'])} labels")
        if system.get("using_sample"):
            st.info("Running on a small synthetic DEMO dataset (not real medical data).")
    else:
        st.warning("Dataset not detected")
        st.caption("Copy Training.csv into Data/ to enable ML features.")
    st.markdown("<div style='border-left:3px solid #fbbf24;background:rgba(120,75,10,.16);padding:12px;border-radius:0 10px 10px 0;color:#fde68a;font-size:.85rem'>Educational use only. Not a diagnosis or a substitute for a clinician.</div>", unsafe_allow_html=True)

def chart_theme(fig, height=360):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(16,26,43,.45)", font_color="#cbd5e1",
        margin=dict(l=20,r=20,t=40,b=20), height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    return fig

def get_symptom_options():
    if isinstance(system, dict) and "error" not in system:
        return [s.replace("_", " ").title() for s in system["symptoms"]]
    return ["Fever","Cough","Sore throat","Headache","Nausea","Vomiting","Diarrhea","Stomach pain","Chest pain","Difficulty breathing","Fainting","Confusion","Severe allergic reaction","Rash","Dizziness","Fatigue","Muscle aches","Painful urination","Back pain","Joint pain"]

if page == "Overview":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">A fresh approach to digital wellness</div>
      <h1>Better questions.<br>Clearer next steps.</h1>
      <p>PULSE PILOT AI combines a clean health-education experience, a dataset-based ML demonstration, explainable model insights, optional generative AI, and downloadable summaries in one modern workspace.</p>
    </div>""", unsafe_allow_html=True)
    cols = st.columns(4)
    cards = [
        ("01","Symptom Navigator","Select symptoms, review safety guidance, and optionally view model outputs."),
        ("02","Model Lab","Compare Decision Tree, Random Forest, and SVM using a held-out split."),
        ("03","AI Companion","Ask general health questions when an API key is configured."),
        ("04","PDF Summary","Download a record of the entered symptoms and model output."),
    ]
    for col,(n,title,desc) in zip(cols,cards):
        with col:
            st.markdown(f'<div class="panel"><div class="metric">{n}</div><h3>{title}</h3><div class="muted">{desc}</div></div>',unsafe_allow_html=True)
    st.write("")
    if isinstance(system, dict) and "error" not in system:
        m=system["metrics"]
        a,b,c,d=st.columns(4)
        a.metric("Unique rows",f"{system['rows']:,}")
        b.metric("Symptoms",len(system["symptoms"]))
        c.metric("Disease labels",len(system["diseases"]))
        d.metric("Best holdout accuracy",f"{max(x['accuracy'] for x in m.values())*100:.1f}%")
        st.caption(f"Metrics are measured on a held-out split after removing {system['duplicates_removed']:,} duplicate rows ({system['rows']:,} unique rows kept). They are not evidence of clinical validity.")
        metric_df=pd.DataFrame([{"Model":k,"Accuracy":v["accuracy"],"Precision":v["precision"],"Recall":v["recall"],"F1":v["f1"]} for k,v in m.items()])
        fig=px.bar(metric_df.melt(id_vars="Model",var_name="Metric",value_name="Score"),x="Model",y="Score",color="Metric",barmode="group",title="Model benchmark")
        st.plotly_chart(chart_theme(fig,340),width="stretch")
    else:
        st.warning("Machine-learning features are waiting for Data/Training.csv. The rest of the app can still be explored.")
        st.code("pulse-pilot-ai/\n├── app.py\n├── health_engine.py\n└── Data/\n    └── Training.csv")

elif page == "Symptom Navigator":
    st.markdown('<div class="eyebrow">Guided check-in</div>',unsafe_allow_html=True)
    st.title("Symptom Navigator")
    st.write("Organize symptoms and review sensible next steps. Model outputs are an educational demonstration, not a diagnosis.")
    with st.form("symptom_form"):
        selected=st.multiselect("What are you experiencing?",get_symptom_options(),placeholder="Search and select symptoms...")
        duration=st.selectbox("Duration",["Started today","1–3 days","4–7 days","More than a week","Recurring"])
        severity=st.select_slider("How severe does it feel?",options=["Mild","Moderate","Severe"],value="Mild")
        worsening=st.radio("Is it getting worse?",["No","Not sure","Yes"],horizontal=True)
        age_group=st.selectbox("Age group (optional)",["Prefer not to say","Under 18","18–64","65+"])
        model_choice=st.selectbox("Model for educational candidates",["Random Forest","SVM","Decision Tree"])
        submitted=st.form_submit_button("Review symptoms",width="stretch")
    if submitted:
        result=assess_symptoms(selected,duration,severity,worsening,age_group)
        st.session_state["last_report"]={"symptoms":selected,"duration":duration,"severity":severity,"worsening":worsening,"result":result,"predictions":[]}
        if result["urgent"]:
            st.error("Urgent warning signs selected")
        elif result["same_day"]:
            st.warning("Consider prompt medical advice")
        else:
            st.success("Check-in summary ready")
        st.markdown(f'<div class="panel"><h3>{result["heading"]}</h3><p>{result["summary"]}</p><b>Suggested next step</b><p>{result["next_step"]}</p></div>',unsafe_allow_html=True)
        if result["urgent"]:
            st.error("If these symptoms are happening now, especially severe breathing difficulty, chest pain, fainting, confusion, or a severe allergic reaction, contact local emergency services immediately.")
        if isinstance(system,dict) and "error" not in system and selected:
            norm={s.replace("_"," ").title():s for s in system["symptoms"]}
            model_syms=[norm[s] for s in selected if s in norm]
            if model_syms:
                st.markdown("### Dataset model candidates")
                st.caption(f"Using {model_choice}. Relative model scores can be misleading and are not calibrated medical probabilities.")
                preds=system["models"][model_choice].predict_candidates(model_syms,3)
                st.session_state["last_report"]["predictions"]=preds
                for p in preds:
                    st.markdown(f"**{p['label']}**")
                    st.progress(float(min(max(p["score"],0),1)),text=f"Model score: {p['score']:.1%}")
        st.caption("Seek professional advice for persistent, worsening, or concerning symptoms.")

elif page == "Model Lab":
    st.markdown('<div class="eyebrow">Transparent machine learning</div>',unsafe_allow_html=True)
    st.title("Model Lab")
    if not isinstance(system,dict) or "error" in system:
        st.error("Cannot load the model dataset. Check that Data/Training.csv exists and includes a prognosis column.")
        if isinstance(system,dict): st.code(system.get("error","Unknown error"))
    else:
        metric_rows=[]
        for name,met in system["metrics"].items():
            row={"Model":name,"Accuracy":met["accuracy"],"Precision (weighted)":met["precision"],"Recall (weighted)":met["recall"],"F1 (weighted)":met["f1"]}
            if met.get("cv_mean") is not None: row[f"{system['cv_folds']}-fold CV accuracy"]=met["cv_mean"]
            metric_rows.append(row)
        df=pd.DataFrame(metric_rows)
        cols=st.columns(4)
        best=max(system["metrics"],key=lambda k:system["metrics"][k]["f1"])
        cols[0].metric("Models compared",len(system["metrics"]))
        cols[1].metric("Best F1",best)
        cols[2].metric("Unique rows",f"{system['rows']:,}")
        cols[3].metric("Feature count",len(system["symptoms"]))
        st.info(f"Data quality: {system['duplicates_removed']:,} duplicate rows were removed before the train/test split to prevent data leakage, so these scores are more honest than raw-dataset numbers. Hold-out set: {system['holdout_rows']} rows.")
        st.dataframe(df.style.format({c:"{:.3f}" for c in df.columns if c!="Model"}),width="stretch",hide_index=True)
        tab1,tab2,tab3=st.tabs(["Metric comparison","Feature importance","Confusion matrix"])
        with tab1:
            melted=df.melt(id_vars="Model",var_name="Metric",value_name="Score")
            fig=px.bar(melted,x="Model",y="Score",color="Metric",barmode="group",title="Held-out evaluation")
            fig.update_yaxes(range=[0,1.05])
            st.plotly_chart(chart_theme(fig),width="stretch")
        with tab2:
            fi=system["feature_importance"]
            fig=px.bar(fi.head(20).sort_values("Importance"),x="Importance",y="Feature",orientation="h",title="Top 20 Random Forest features")
            st.plotly_chart(chart_theme(fig,600),width="stretch")
            st.caption("Feature importance shows how the fitted model used features in this dataset; it does not establish medical causation.")
        with tab3:
            chosen=st.selectbox("Choose model",list(system["confusion"]))
            cm=system["confusion"][chosen]
            labels=system["class_labels"]
            fig=go.Figure(data=go.Heatmap(z=cm,x=labels,y=labels,colorscale="Teal",hoverongaps=False))
            fig.update_layout(title=f"{chosen} confusion matrix",xaxis_title="Predicted label",yaxis_title="True label")
            st.plotly_chart(chart_theme(fig,650),width="stretch")
            st.caption("Large class counts can make the confusion matrix dense. Use it as a model-evaluation view, not a clinical validation result.")

elif page == "AI Health Chat":
    st.markdown('<div class="eyebrow">Ask in plain language</div>',unsafe_allow_html=True)
    st.title("AI Health Companion")
    st.write("General health education, plain-language explanations, and help preparing questions for a clinician.")
    key_present=bool(os.getenv("OPENAI_API_KEY"))
    try: key_present=key_present or bool(st.secrets.get("OPENAI_API_KEY",""))
    except Exception: pass
    if key_present: st.success("AI API key detected.")
    else: st.info("Live answers require an API key. The rest of PULSE PILOT AI works without it. See README.md.")
    if "chat_messages" not in st.session_state: st.session_state.chat_messages=[]
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    prompt=st.chat_input("Ask a general health question...")
    if prompt:
        user_turns=sum(1 for m in st.session_state.chat_messages if m["role"]=="user")
        if user_turns>=MAX_CHAT_TURNS:
            st.warning(f"Session limit of {MAX_CHAT_TURNS} questions reached. Refresh the page to start a new session.")
        else:
            history=list(st.session_state.chat_messages)
            st.session_state.chat_messages.append({"role":"user","content":prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                if looks_like_emergency(prompt):
                    st.error(emergency_warning())
                with st.spinner("Preparing a careful response..."): answer=ask_ai(prompt,history)
                st.markdown(answer)
            st.session_state.chat_messages.append({"role":"assistant","content":answer})
    st.caption("Do not share identifying information. AI responses can be wrong and must not be used for emergency decisions.")

elif page == "Health Library":
    st.markdown('<div class="eyebrow">Health information, without the jargon</div>',unsafe_allow_html=True)
    st.title("Health Library")
    query=st.text_input("Search topics",placeholder="Try fever, hydration, antibiotics...")
    for topic in get_health_topics():
        if query.lower() in (topic["title"]+" "+topic["text"]).lower():
            with st.expander(topic["title"],expanded=bool(query)):
                st.write(topic["text"])
                st.caption(topic["note"])

elif page == "PDF Report":
    st.markdown('<div class="eyebrow">Takeaway summary</div>',unsafe_allow_html=True)
    st.title("Download a PDF summary")
    report=st.session_state.get("last_report")
    if not report:
        st.info("First use Symptom Navigator and click Review symptoms. Then return here to generate a report.")
    else:
        st.write("Review the current session summary before downloading.")
        st.write("**Symptoms:**",", ".join(report["symptoms"]) if report["symptoms"] else "None selected")
        st.write("**Duration:**",report["duration"]," · **Severity:**",report["severity"])
        st.write("**Suggested next step:**",report["result"]["next_step"])
        if report["predictions"]:
            st.write("**Dataset model candidates:**")
            for p in report["predictions"]: st.write(f"- {p['label']}: {p['score']:.1%} model score")
        try:
            pdf=build_pdf_report(report)
            st.download_button("📄 Download PDF summary",data=pdf,file_name="Pulse_Pilot_AI_Symptom_Summary.pdf",mime="application/pdf",width="stretch")
        except Exception as e:
            st.error(f"Could not create PDF: {e}")
            st.caption("Ensure reportlab is installed from requirements.txt.")

elif page == "About":
    st.markdown(
        '<div class="eyebrow">PROJECT OVERVIEW</div>',
        unsafe_allow_html=True
    )

    st.title("About PULSE PILOT AI")

    st.markdown("""
    ### 🩺 AI-Powered Health Information & Symptom Analysis

    PULSE PILOT AI is an intelligent health companion designed to help
    users explore their symptoms, understand common health conditions,
    and access health information through an interactive web application.

    The project combines machine learning, data visualization, and
    AI-assisted conversations to create an accessible and informative
    digital health experience.
    """)

    st.divider()

    st.subheader("🎯 Project Objectives")

    st.markdown("""
    - Provide an interactive platform for exploring common symptoms.
    - Use machine learning to identify potential condition patterns
      from selected symptoms.
    - Present model performance through clear metrics and visualizations.
    - Explain model behavior using feature-importance analysis.
    - Offer an AI-assisted health information chat interface.
    - Generate downloadable PDF summaries of symptom reviews.
    - Promote awareness of health information and responsible use of AI.
    """)

    st.divider()

    st.subheader("⚙️ Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 🤖 Machine Learning
        - Decision Tree classification
        - Random Forest classification
        - Support Vector Machine (SVM)
        - Model performance comparison

        #### 🩺 Symptom Navigator
        - Interactive symptom selection
        - Symptom review and guidance
        - Educational condition suggestions
        """)

    with col2:
        st.markdown("""
        #### 📊 Model Explainability
        - Accuracy and evaluation metrics
        - Confusion matrix visualization
        - Feature-importance analysis

        #### 📄 PDF Reports
        - Symptom review summaries
        - Downloadable PDF documents
        - Organized information for later reference
        """)

    st.divider()

    st.subheader("🛠️ Technology Stack")

    technologies = [
        ("Python", "Core programming language"),
        ("Streamlit", "Interactive web application"),
        ("Pandas & NumPy", "Data processing and numerical operations"),
        ("Scikit-learn", "Machine learning and model evaluation"),
        ("Plotly", "Interactive charts and visualizations"),
        ("ReportLab", "PDF report generation"),
        ("OpenAI API", "Optional generative AI integration"),
    ]

    for technology, description in technologies:
        st.markdown(f"**{technology}** · {description}")

    st.divider()

    st.subheader("🌟 Project Impact")

    st.write("""
    PULSE PILOT AI demonstrates how machine learning and modern web
    technologies can be combined to make health information more
    accessible and understandable. It brings symptom exploration,
    model evaluation, interactive information, and report generation
    together in one application.
    """)

    st.info(
        "PULSE PILOT AI is an educational project, not a medical "
        "diagnostic system. Its predictions and AI-generated responses "
        "may be inaccurate and should not replace professional medical "
        "advice, diagnosis, or treatment."
    )
