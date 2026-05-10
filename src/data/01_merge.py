import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = Path(BASE_DIR/'data'/'raw')
PROC_DIR = Path(BASE_DIR/'data'/'processed')
PROC_DIR.mkdir(parents=True, exist_ok=True)


# load notes
notes = pd.read_csv(
    RAW_DIR/'NOTEEVENTS.csv',
    usecols=['SUBJECT_ID', 'HADM_ID', 'CATEGORY', 'ISERROR', 'TEXT'],
    low_memory=False
)

notes = notes[notes['CATEGORY'] == 'Discharge summary']
notes = notes[notes['ISERROR'].isna() | (notes['ISERROR'] != 1)]
notes = notes[['SUBJECT_ID', 'HADM_ID', 'TEXT']].copy()


# load diagnoses
diagnoses = pd.read_csv(
    RAW_DIR/'DIAGNOSES_ICD.csv',
    usecols=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE'],
    low_memory=False
)
diagnoses = diagnoses.dropna(subset=['ICD9_CODE'])

icd = (
    diagnoses
    .sort_values(['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'])
    .groupby(['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE']
    .apply(list)
    .reset_index()
)


# load admissions
admissions = pd.read_csv(
    RAW_DIR/'ADMISSIONS.csv',
    usecols=['SUBJECT_ID', 'HADM_ID', 'HOSPITAL_EXPIRE_FLAG', 'DISCHTIME'],
    low_memory=False
)


# load patients
patients = pd.read_csv(
    RAW_DIR/'PATIENTS.csv',
    usecols=['SUBJECT_ID', 'GENDER', 'DOB'],
    low_memory=False
)


# merge tables
df = notes.merge(icd, on=['SUBJECT_ID', 'HADM_ID'], how='left')
df = df.merge(admissions, on=['SUBJECT_ID', 'HADM_ID'], how='left')
df = df.merge(patients, on='SUBJECT_ID', how='left')

df['ICD9_CODE'] = df['ICD9_CODE'].apply(
    lambda codes: '|'.join(map(str, codes)) if isinstance(codes, list) else ''
)

df.drop_duplicates(subset=['SUBJECT_ID', 'HADM_ID'], inplace=True)
df.reset_index(drop=True, inplace=True)


# save merged data
cols = ['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE',
        'HOSPITAL_EXPIRE_FLAG', 'GENDER', 'DOB', 'DISCHTIME', 'TEXT']
df.to_csv(PROC_DIR/'merged_data.csv', index=False, columns=cols)
print('merged.')