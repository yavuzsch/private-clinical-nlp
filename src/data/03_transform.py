import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = Path(BASE_DIR/'data'/'raw')
PROC_DIR = Path(BASE_DIR/'data'/'processed')


# load cleaned data
df = pd.read_csv(PROC_DIR/'cleaned_data.csv', low_memory=False)


# load ICD reference for validation
d_icd = pd.read_csv(RAW_DIR/'D_ICD_DIAGNOSES.csv', usecols=['ICD9_CODE'], low_memory=False)
valid_icd = set(d_icd['ICD9_CODE'].astype(str).str.strip())


# filter invalid ICD codes
def filter_valid_icd(codes):
    kept = [c.strip() for c in codes.split('|') if c.strip() in valid_icd]
    return '|'.join(kept)

df['ICD9_CODE'] = df['ICD9_CODE'].apply(filter_valid_icd)
df = df[df['ICD9_CODE'].str.len() > 0].reset_index(drop=True)
print('ICD validation done.')


# compute age bucket
df['DISCHTIME'] = pd.to_datetime(df['DISCHTIME'], errors='coerce')
df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')


def age_bucket(row):
    dob = row['DOB']
    disch = row['DISCHTIME']

    if pd.isnull(dob) or pd.isnull(disch) or dob.year < 1900 or disch.year < 1900:
        return 'UNK'

    try:
        age = min((disch.to_pydatetime() - dob.to_pydatetime()).days // 365, 90)
    except OverflowError:
        return 'UNK'

    if age < 30: return '<30'
    if age < 50: return '30-49'
    if age < 70: return '50-69'
    return '70+'

df['AGE_BUCKET'] = df.apply(age_bucket, axis=1)
df.drop(columns=['DOB', 'DISCHTIME'], inplace=True)


# save metadata and transformed data
metadata_cols = ['SUBJECT_ID', 'HADM_ID', 'GENDER', 'AGE_BUCKET', 'HOSPITAL_EXPIRE_FLAG']
df[metadata_cols].to_csv(PROC_DIR/'metadata.csv', index=False)
print('metadata saved.')

cols = ['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE', 'TEXT']
df.to_csv(PROC_DIR/'transformed_data.csv', index=False, columns=cols)
print('transformed.')