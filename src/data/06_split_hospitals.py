import json
import numpy as np
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROC_DIR = Path(BASE_DIR/'data'/'processed')
SPL_DIR = Path(BASE_DIR/'data'/'split')

SEED = 42
NUM_HOSPITALS = 10

np.random.seed(SEED)


def split_hospitals(df):
    unique_pts = df['SUBJECT_ID'].unique().copy()
    np.random.shuffle(unique_pts)

    chunks = np.array_split(unique_pts, NUM_HOSPITALS)

    partitions = {}
    for h_id, chunk in enumerate(chunks):
        part = df[df['SUBJECT_ID'].isin(chunk)].copy()
        part = part.drop_duplicates(subset=['SUBJECT_ID', 'HADM_ID']).reset_index(drop=True)
        part['ICD_CATEGORIES'] = part['ICD_CATEGORIES'].apply(
            lambda x: '|'.join(x) if isinstance(x, list) else x
        )
        partitions[h_id] = part

    return partitions


# load train data
df = pd.read_csv(SPL_DIR/'train.csv')
df['ICD_CATEGORIES'] = df['ICD_CATEGORIES'].apply(lambda x: x.split('|'))

with open(PROC_DIR/'icd_category_meta.json') as f:
    meta = json.load(f)


# split hospitals IID
partitions = split_hospitals(df)
summary = {'hospitals': {}}

for h_id, part in partitions.items():
    hosp_dir = SPL_DIR/f'hospital_{h_id}'
    hosp_dir.mkdir(parents=True, exist_ok=True)

    part.to_csv(hosp_dir/'train.csv', index=False)

    info = {
        'hospital_id': h_id,
        'total': len(part),
        'unique_patients': int(part['SUBJECT_ID'].nunique()),
    }
    with open(hosp_dir/'info.json', 'w') as f:
        json.dump(info, f, indent=2)

    summary['hospitals'][h_id] = info
    print(f'hospital_{h_id}: train={len(part):,}')


# save split hospitals summary
summary['seed'] = SEED
summary['num_hospitals'] = NUM_HOSPITALS
summary['categories'] = meta['categories']
summary['num_labels'] = meta['num_labels']

with open(SPL_DIR/'split_hospitals_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('split hospitals.')
print('num labels:', meta['num_labels'])