import json
import pandas as pd
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROC_DIR = Path(BASE_DIR/'data'/'processed')
MIN_FREQ = 100


def icd9_to_category(code):
    code = str(code).strip().upper()
    if not code:
        return None

    if code[0] == 'V':
        return 'supplementary_V'
    if code[0] == 'E':
        return 'supplementary_E'
    if not code[0].isdigit():
        return None

    try:
        num = int(''.join(filter(str.isdigit, code[:3])) or '0')
    except ValueError:
        return None

    if 1 <= num <= 139: return 'infectious'
    elif 140 <= num <= 239: return 'neoplasm'
    elif 240 <= num <= 279: return 'endocrine'
    elif 280 <= num <= 289: return 'blood'
    elif 290 <= num <= 319: return 'mental'
    elif 320 <= num <= 389: return 'nervous'
    elif 390 <= num <= 459: return 'circulatory'
    elif 460 <= num <= 519: return 'respiratory'
    elif 520 <= num <= 579: return 'digestive'
    elif 580 <= num <= 629: return 'genitourinary'
    elif 630 <= num <= 679: return 'pregnancy'
    elif 680 <= num <= 709: return 'skin'
    elif 710 <= num <= 739: return 'musculoskeletal'
    elif 740 <= num <= 759: return 'congenital'
    elif 760 <= num <= 779: return 'perinatal'
    elif 780 <= num <= 799: return 'symptoms'
    elif 800 <= num <= 999: return 'injury'
    return None


def codes_to_categories(raw):
    cats = []
    seen = set()
    for code in str(raw).split('|'):
        cat = icd9_to_category(code.strip())
        if cat and cat not in seen:
            seen.add(cat)
            cats.append(cat)
    return cats


# load transformed data
df = pd.read_csv(PROC_DIR/'transformed_data.csv')


# map to categories
df['ICD_CATEGORIES'] = df['ICD9_CODE'].fillna('').apply(codes_to_categories)


# count frequencies
all_cats = [c for cats in df['ICD_CATEGORIES'] for c in cats]
cat_freq = Counter(all_cats)

print('total categories:', len(cat_freq))


# filter rare categories
valid_cats = {cat for cat, cnt in cat_freq.items() if cnt >= MIN_FREQ}
print('valid categories:', len(valid_cats))

df['ICD_CATEGORIES'] = df['ICD_CATEGORIES'].apply(
    lambda cats: [c for c in cats if c in valid_cats]
)

df = df[df['ICD_CATEGORIES'].map(len) > 0].reset_index(drop=True)
df['ICD_CATEGORIES'] = df['ICD_CATEGORIES'].apply('|'.join)


# save category metadata and filtered data
sorted_cats = sorted(valid_cats)
meta = {
    'categories': sorted_cats,
    'cat_to_idx': {cat: i for i, cat in enumerate(sorted_cats)},
    'num_labels': len(sorted_cats),
    'min_freq': MIN_FREQ,
}

with open(PROC_DIR/'icd_category_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

cols = ['SUBJECT_ID', 'HADM_ID', 'ICD_CATEGORIES', 'TEXT']
df.to_csv(PROC_DIR/'filtered_data.csv', index=False, columns=cols)
print('filtered.')