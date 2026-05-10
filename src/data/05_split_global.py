import json
import numpy as np
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROC_DIR = Path(BASE_DIR/'data'/'processed')
SPL_DIR = Path(BASE_DIR/'data'/'split')
SPL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
VAL_RATIO = 0.15
TEST_RATIO = 0.15

np.random.seed(SEED)


def split_global(df):
    unique_pts = df['SUBJECT_ID'].unique().copy()
    np.random.shuffle(unique_pts)

    n = len(unique_pts)
    n_test = int(n * TEST_RATIO)
    n_val = int(n * VAL_RATIO)

    test_pts = set(unique_pts[:n_test])
    val_pts = set(unique_pts[n_test:n_test + n_val])
    train_pts = set(unique_pts[n_test + n_val:])

    train = df[df['SUBJECT_ID'].isin(train_pts)].reset_index(drop=True)
    val = df[df['SUBJECT_ID'].isin(val_pts)].reset_index(drop=True)
    test = df[df['SUBJECT_ID'].isin(test_pts)].reset_index(drop=True)

    return train, val, test


# load filtered data
df = pd.read_csv(PROC_DIR/'filtered_data.csv')


# split global patient-level
train, val, test = split_global(df)
print(f'train={len(train):,} | val={len(val):,} | test={len(test):,}')


# save split global data
train.to_csv(SPL_DIR/'train.csv', index=False)
val.to_csv(SPL_DIR/'val.csv', index=False)
test.to_csv(SPL_DIR/'test.csv', index=False)

info = {
    'seed': SEED,
    'train': len(train),
    'val': len(val),
    'test': len(test),
}
with open(SPL_DIR/'split_global_info.json', 'w') as f:
    json.dump(info, f, indent=2)

print('split global.')