import json
import pickle
import argparse
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--max_length', type=int, default=256)
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROC_DIR = Path(BASE_DIR/'data'/'processed')
SPL_DIR = Path(BASE_DIR/'data'/'split')
MODEL_SLUG = args.model_name.replace('/', '_')


# load metadata
with open(PROC_DIR/'icd_category_meta.json') as f:
    meta = json.load(f)

with open(SPL_DIR/'split_hospitals_summary.json') as f:
    summary = json.load(f)

NUM_HOSPITALS = summary['num_hospitals']
ALL_CATEGORIES = meta['categories']
NUM_LABELS = meta['num_labels']


# tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
tokenizer.truncation_side = 'left'


# fit MLB on all hospital train sets
print('fitting MultiLabelBinarizer...')
all_train_labels = []

for h_id in range(NUM_HOSPITALS):
    train_df = pd.read_csv(SPL_DIR/f'hospital_{h_id}'/'train.csv')
    train_df['ICD_CATEGORIES'] = train_df['ICD_CATEGORIES'].apply(lambda x: x.split('|'))
    all_train_labels.extend(train_df['ICD_CATEGORIES'].tolist())

mlb = MultiLabelBinarizer(classes=ALL_CATEGORIES)
mlb.fit(all_train_labels)
print(f'classes: {len(mlb.classes_)}')


def build_dataset(split_df, mlb):
    split_df = split_df.reset_index(drop=True)
    Y = mlb.transform(split_df['ICD_CATEGORIES'])

    ds = Dataset.from_pandas(split_df[['TEXT']])
    ds = ds.map(
        lambda batch: tokenizer(
            batch['TEXT'],
            truncation=True,
            padding='max_length',
            max_length=args.max_length,
        ),
        batched=True
    )
    ds = ds.add_column('labels', [[float(v) for v in row] for row in Y])
    keep = ['input_ids', 'attention_mask', 'labels']
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    ds.set_format(type='torch', columns=keep)
    return ds


# tokenize global val and test
print('\ntokenizing global val and test...')
global_out = SPL_DIR/MODEL_SLUG
global_out.mkdir(parents=True, exist_ok=True)

for split in ['val', 'test']:
    split_df = pd.read_csv(SPL_DIR/f'{split}.csv')
    split_df['ICD_CATEGORIES'] = split_df['ICD_CATEGORIES'].apply(lambda x: x.split('|'))
    ds = build_dataset(split_df, mlb)
    ds.save_to_disk(str(global_out/split))
    print(f'{split}: {len(ds):,}')


# tokenize global train (centralized baseline)
print('\ntokenizing global train...')
global_train_df = pd.read_csv(SPL_DIR/'train.csv')
global_train_df['ICD_CATEGORIES'] = global_train_df['ICD_CATEGORIES'].apply(lambda x: x.split('|'))
ds = build_dataset(global_train_df, mlb)
ds.save_to_disk(str(global_out/'train'))
print(f'global train: {len(ds):,}')


# tokenize local train per hospital
for h_id in range(NUM_HOSPITALS):
    print(f'\ntokenizing hospital_{h_id} train...')

    hosp_dir = SPL_DIR/f'hospital_{h_id}'
    out_dir = hosp_dir/MODEL_SLUG
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(hosp_dir/'train.csv')
    train_df['ICD_CATEGORIES'] = train_df['ICD_CATEGORIES'].apply(lambda x: x.split('|'))
    ds = build_dataset(train_df, mlb)
    ds.save_to_disk(str(out_dir/'train'))
    print(f'train: {len(ds):,}')

    with open(out_dir/'label_info.json', 'w') as f:
        json.dump({
            'model_name': args.model_name,
            'max_length': args.max_length,
            'num_labels': NUM_LABELS,
            'categories': ALL_CATEGORIES,
        }, f, indent=2)


# save global mlb
with open(SPL_DIR/f'global_mlb_{MODEL_SLUG}.pkl', 'wb') as f:
    pickle.dump(mlb, f)

with open(global_out/'label_info.json', 'w') as f:
    json.dump({
        'model_name': args.model_name,
        'max_length': args.max_length,
        'num_labels': NUM_LABELS,
        'categories': ALL_CATEGORIES,
    }, f, indent=2)

print('\ntokenized.')
print(f'num labels: {NUM_LABELS}')