import re
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROC_DIR = Path(BASE_DIR/'data'/'processed')


# load merged data
df = pd.read_csv(PROC_DIR/'merged_data.csv', low_memory=False)


# sanitize text
def sanitize_text(text):
    if pd.isna(text):
        return ''

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

df['TEXT'] = df['TEXT'].apply(sanitize_text)


# aggregate duplicate admissions
df = (
    df.groupby(['SUBJECT_ID', 'HADM_ID'], as_index=False)
    .agg({
        'TEXT': '\n\n'.join,
        'ICD9_CODE': 'first',
        'HOSPITAL_EXPIRE_FLAG': 'first',
        'GENDER': 'first',
        'DOB': 'first',
        'DISCHTIME': 'first',
    })
)


# drop empty records
df['ICD9_CODE'] = df['ICD9_CODE'].fillna('')
df = df[df['ICD9_CODE'].str.len() > 0]
df = df[df['TEXT'].str.strip().str.len() > 50].reset_index(drop=True)


# save cleaned data
cols = ['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE',
        'HOSPITAL_EXPIRE_FLAG', 'GENDER', 'DOB', 'DISCHTIME', 'TEXT']
df.to_csv(PROC_DIR/'cleaned_data.csv', index=False, columns=cols)
print('cleaned.')