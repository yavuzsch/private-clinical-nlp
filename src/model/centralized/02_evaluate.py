import json
import argparse
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    classification_report,
)
import torch


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SPL_DIR = BASE_DIR/'data'/'split'
MODEL_DIR = BASE_DIR/'models'/'centralized'/args.model_name.replace('/', '_')
MODEL_SLUG = args.model_name.replace('/', '_')


# load metadata
with open(BASE_DIR/'data'/'processed'/'icd_category_meta.json') as f:
    meta = json.load(f)

NUM_LABELS = meta['num_labels']
CATEGORIES = meta['categories']

print(f'model : {args.model_name}')
print(f'num_labels : {NUM_LABELS}')


# load test dataset
print('loading test dataset...')
test_ds = load_from_disk(str(SPL_DIR/MODEL_SLUG/'test'))
print(f'test={len(test_ds):,}')


# load model
print('loading model...')
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR/'best_model'))


# run inference
print('running inference...')
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir=str(MODEL_DIR/'tmp'), report_to='none'),
)
predictions = trainer.predict(test_ds)

probs = torch.sigmoid(torch.tensor(predictions.predictions)).numpy()
preds = (probs >= 0.5).astype(int)
labels = predictions.label_ids


# compute metrics
f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
accuracy = accuracy_score(labels, preds)

try:
    auc = roc_auc_score(labels, probs, average='macro', multi_class='ovr')
except ValueError:
    auc = 0.0

print(f'test results:')
print(f'  f1_macro : {f1_macro:.4f}')
print(f'  f1_weighted : {f1_weighted:.4f}')
print(f'  accuracy : {accuracy:.4f}')
print(f'  auc : {auc:.4f}')


# per-class report
print('\nper-class report:')
print(classification_report(labels, preds, target_names=CATEGORIES, zero_division=0))


# save results
results = {
    'model_name': args.model_name,
    'num_labels': NUM_LABELS,
    'test': {
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'accuracy': accuracy,
        'auc': auc,
    }
}
with open(MODEL_DIR/'test_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'done.')