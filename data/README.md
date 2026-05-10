# Data Pipeline
Preprocesses MIMIC-III clinical notes and prepares federated datasets for 10 virtual hospitals.


## Requirements
Download the following tables from [PhysioNet](https://physionet.org/content/mimiciii/1.4/) and place them in `data/raw/`:

```
data/raw/
├── ADMISSIONS.csv
├── D_ICD_DIAGNOSES.csv
├── DIAGNOSES_ICD.csv
├── NOTEEVENTS.csv
└── PATIENTS.csv
```

Access requires completing the CITI training and signing the data use agreement on PhysioNet.


## Setup
```bash
pip install numpy pandas scikit-learn datasets transformers torch
```


## Usage
Run scripts in order from the project root:

```bash
python src/data/01_merge.py
python src/data/02_clean.py
python src/data/03_transform.py
python src/data/04_filter_icd.py
python src/data/05_split_global.py
python src/data/06_split_hospitals.py
python src/data/07_tokenize.py --model_name distilbert-base-uncased --max_length 256
```

To tokenize for multiple models:

```bash
python src/data/07_tokenize.py --model_name bert-base-uncased --max_length 256
python src/data/07_tokenize.py --model_name emilyalsentzer/Bio_ClinicalBERT --max_length 256
```


## Structure
```
private-clinical-nlp/
├── data/
│   ├── raw/  # not included, see Requirements
│   ├── processed/  # not included, see Usage
│   │   ├── merged_data.csv
│   │   ├── cleaned_data.csv
│   │   ├── transformed_data.csv
│   │   ├── metadata.csv
│   │   ├── filtered_data.csv
│   │   └── icd_category_meta.json
│   └── split/  # not included, see Usage
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       ├── split_global_info.json
│       ├── split_hospitals_summary.json
│       ├── global_mlb_<model_slug>.pkl
│       ├── <model_slug>/
│       │   ├── train/  # centralized baseline
│       │   ├── val/
│       │   ├── test/
│       │   └── label_info.json
│       ├── hospital_0/
│       │   ├── train.csv
│       │   ├── info.json
│       │   └── <model_slug>/
│       │       ├── train/
│       │       └── label_info.json
│       ├── hospital_1/  # same structure as hospital_0
│       ├── hospital_2/  # same structure as hospital_0
│       ├── hospital_3/  # same structure as hospital_0
│       ├── hospital_4/  # same structure as hospital_0
│       ├── hospital_5/  # same structure as hospital_0
│       ├── hospital_6/  # same structure as hospital_0
│       ├── hospital_7/  # same structure as hospital_0
│       ├── hospital_8/  # same structure as hospital_0
│       └── hospital_9/  # same structure as hospital_0
└── src/
    └── data/
        ├── 01_merge.py
        ├── 02_clean.py
        ├── 03_transform.py
        ├── 04_filter_icd.py
        ├── 05_split_global.py
        ├── 06_split_hospitals.py
        └── 07_tokenize.py
```


## Hospital Distribution
Train data is split randomly and equally across 10 hospitals using IID distribution. Splits are patient-level to prevent data leakage.