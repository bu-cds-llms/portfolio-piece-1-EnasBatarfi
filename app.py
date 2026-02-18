from pathlib import Path
import joblib
import streamlit as st
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Prompt Injection Detector", layout="centered")

PROJECT_ROOT = Path(__file__).resolve().parents[0]
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

BASELINE_DIR = MODELS_DIR / "baseline_tfidf_logreg"
TRANS_DIR = MODELS_DIR / "distilbert_prompt_injection"

st.title("Prompt Injection Detector")
st.write("Pick a model, paste a prompt, get a probability of injection.")

@st.cache_resource
def load_baseline():
    tfidf = joblib.load(BASELINE_DIR / "tfidf.joblib")
    clf = joblib.load(BASELINE_DIR / "logreg.joblib")
    return tfidf, clf

@st.cache_resource
def load_transformer():
    tokenizer = AutoTokenizer.from_pretrained(str(TRANS_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(TRANS_DIR))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

def predict_baseline(text: str):
    tfidf, clf = load_baseline()
    X = tfidf.transform([text])
    p_inj = clf.predict_proba(X)[0, 1]
    return float(p_inj)

def predict_transformer(text: str):
    tokenizer, model, device = load_transformer()
    enc = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
    return float(probs[1])

model_choice = st.selectbox(
    "Model",
    ["Transformer (fine-tuned DistilBERT)", "Baseline (TF-IDF + LogReg)"]
)

threshold = st.slider("Decision threshold", 0.0, 1.0, 0.7, 0.01)

text = st.text_area("Prompt", height=200, placeholder="Paste your prompt here...")

col1, col2 = st.columns(2)
with col1:
    run = st.button("Check")
with col2:
    st.button("Clear", on_click=lambda: st.session_state.update({"_": None}))

if run:
    if not text.strip():
        st.warning("Paste a prompt first.")
    else:
        if model_choice.startswith("Transformer"):
            p = predict_transformer(text)
        else:
            p = predict_baseline(text)

        verdict = "BLOCK (likely injection)" if p >= threshold else "ALLOW"
        st.subheader(verdict)
        st.metric("P(injection)", f"{p:.3f}")
        st.caption("Higher means more likely the prompt contains injection patterns.")
