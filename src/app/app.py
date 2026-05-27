import json
import pickle
import re
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROC_DIR = BASE_DIR/'data'/'processed'
SPL_DIR = BASE_DIR/'data'/'split'
MODEL_SLUG = 'emilyalsentzer_Bio_ClinicalBERT'

st.set_page_config(
    page_title='Clinical NLP',
    page_icon='🏥',
    layout='wide',
)


# load metadata
@st.cache_resource
def load_metadata():
    with open(PROC_DIR/'icd_category_meta.json') as f:
        meta = json.load(f)
    with open(SPL_DIR/f'global_mlb_{MODEL_SLUG}.pkl', 'rb') as f:
        mlb = pickle.load(f)
    return meta, mlb


# load model and tokenizer
@st.cache_resource
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', use_fast=True)
    tokenizer.truncation_side = 'left'
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


# sanitize text
def sanitize_text(text):
    text = str(text).lower()
    text = re.sub(r'\[\*\*.*?\*\*\]', '[PHI]', text, flags=re.DOTALL)
    text = re.sub(r'(date of birth|dob)\s*:.*?(?=\n|service:)', ' ', text)
    text = re.sub(r'\b\d{1,3}[-\s]*(year|yr)[-\s]*old\b', '[AGE]', text)
    text = re.sub(r'\b\d{1,3}\s*(yo|y/o)\b', '[AGE]', text)
    text = re.sub(r'\b(male|female|man|woman|gentleman|lady)\b', '[GENDER]', text)
    text = re.sub(r'\b(he|she|his|her|him)\b', '[GENDER]', text)
    text = re.sub(r'\b\d{6,}\b', '[NUM]', text)
    text = re.sub(r'admission date\s*:.*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'discharge date\s*:.*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# predict
def predict(text, tokenizer, model, mlb, threshold=0.5):
    inputs = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='pt',
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).squeeze().numpy()
    categories = mlb.classes_
    results = sorted(
        [(cat, float(prob)) for cat, prob in zip(categories, probs)],
        key=lambda x: x[1],
        reverse=True,
    )
    predictions = [(cat, prob) for cat, prob in results if prob >= threshold]
    return predictions, results


# model options
MODEL_OPTIONS = {
    'Centralized': str(BASE_DIR/'models'/'centralized'/MODEL_SLUG/'best_model'),
    'Federated': str(BASE_DIR/'models'/'federated'/MODEL_SLUG/'best_model'),
    'Federated + DP (ε=1.0)': str(BASE_DIR/'models'/'federated_dp'/MODEL_SLUG/'epsilon_1.0'/'best_model'),
    'Federated + DP (ε=3.0)': str(BASE_DIR/'models'/'federated_dp'/MODEL_SLUG/'epsilon_3.0'/'best_model'),
    'Federated + DP (ε=8.0)': str(BASE_DIR/'models'/'federated_dp'/MODEL_SLUG/'epsilon_8.0'/'best_model'),
}

EPSILON_MAP = {
    'Centralized': '∞',
    'Federated': '∞',
    'Federated + DP (ε=1.0)': '1.0',
    'Federated + DP (ε=3.0)': '3.0',
    'Federated + DP (ε=8.0)': '8.0',
}


# sidebar
st.sidebar.title('Clinical NLP')
st.sidebar.caption('Privacy-Preserving ICD-9 Classification')

page = st.sidebar.radio('Page', ['Prediction', 'Model Comparison'])

meta, mlb = load_metadata()


# prediction page
if page == 'Prediction':
    st.title('ICD-9 Category Prediction')

    col1, col2 = st.columns([1, 1])

    with col1:
        note = st.text_area('Clinical note', height=200, placeholder='Enter a clinical note...')
        model_choice = st.selectbox('Model', list(MODEL_OPTIONS.keys()))
        threshold = st.slider('Confidence threshold', 0.1, 0.9, 0.5, 0.05)
        predict_btn = st.button('Predict', use_container_width=True)

    with col2:
        if predict_btn and note.strip():
            sanitized = sanitize_text(note)

            with st.expander('Sanitized note (PHI removed)'):
                st.text(sanitized)

            model_path = MODEL_OPTIONS[model_choice]

            with st.spinner('Loading model...'):
                tokenizer, model = load_model(model_path)

            predictions, all_results = predict(sanitized, tokenizer, model, mlb, threshold)

            st.subheader('Predictions')

            if predictions:
                for cat, prob in predictions:
                    st.progress(prob, text=f'{cat}  —  {prob:.2f}')
            else:
                st.info('No category above threshold.')

            epsilon = EPSILON_MAP[model_choice]
            st.caption(f'Model: {model_choice}  |  ε = {epsilon}  |  δ = 1e-5')

        elif predict_btn:
            st.warning('Please enter a clinical note.')


# comparison page
elif page == 'Model Comparison':
    st.title('Model Comparison')

    results = {
        'Centralized': {'f1_macro': 0.5641, 'f1_micro': 0.6779, 'precision_macro': 0.7464, 'recall_macro': 0.5059, 'auc': 0.8169},
        'Federated': {'f1_macro': None, 'f1_micro': None, 'precision_macro': None, 'recall_macro': None, 'auc': None},
        'Federated + DP (ε=1.0)': {'f1_macro': None, 'f1_micro': None, 'precision_macro': None, 'recall_macro': None, 'auc': None},
        'Federated + DP (ε=3.0)': {'f1_macro': None, 'f1_micro': None, 'precision_macro': None, 'recall_macro': None, 'auc': None},
        'Federated + DP (ε=8.0)': {'f1_macro': None, 'f1_micro': None, 'precision_macro': None, 'recall_macro': None, 'auc': None},
    }

    st.subheader('Metrics')

    import pandas as pd
    rows = []
    for model_name, metrics in results.items():
        row = {'Model': model_name, 'ε': EPSILON_MAP[model_name]}
        for k, v in metrics.items():
            row[k] = f'{v:.4f}' if v is not None else '—'
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.caption('Results will be updated after federated and DP training is complete.')