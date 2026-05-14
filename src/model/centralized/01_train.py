import json
import argparse
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    hamming_loss,
)
import torch


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--max_length', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=2e-5)
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SPL_DIR = BASE_DIR/'data'/'split'
MODEL_DIR = BASE_DIR/'models'/'centralized'/args.model_name.replace('/', '_')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SLUG = args.model_name.replace('/', '_')
SEED = 42


# load metadata
with open(BASE_DIR/'data'/'processed'/'icd_category_meta.json') as f:
    meta = json.load(f)

NUM_LABELS = meta['num_labels']
print(f'model : {args.model_name}')
print(f'num_labels : {NUM_LABELS}')


# load datasets
print('loading datasets...')
train_ds = load_from_disk(str(SPL_DIR/MODEL_SLUG/'train'))
val_ds = load_from_disk(str(SPL_DIR/MODEL_SLUG/'val'))
test_ds = load_from_disk(str(SPL_DIR/MODEL_SLUG/'test'))
print(f'train={len(train_ds):,} | val={len(val_ds):,} | test={len(test_ds):,}')


# load model
print('loading model...')
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=NUM_LABELS,
    problem_type='multi_label_classification',
)


# metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
    precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
    precision_micro = precision_score(labels, preds, average='micro', zero_division=0)
    recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
    recall_micro = recall_score(labels, preds, average='micro', zero_division=0)
    hamming = hamming_loss(labels, preds)
    accuracy = accuracy_score(labels, preds)

    try:
        auc = roc_auc_score(labels, probs, average='macro', multi_class='ovr')
    except ValueError:
        auc = 0.0

    return {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'precision_micro': precision_micro,
        'recall_macro': recall_macro,
        'recall_micro': recall_micro,
        'hamming_loss': hamming,
        'accuracy': accuracy,
        'auc': auc,
    }


# training arguments
training_args = TrainingArguments(
    output_dir=str(MODEL_DIR/'checkpoints'),
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size * 2,
    learning_rate=args.learning_rate,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    greater_is_better=True,
    fp16=True,
    seed=SEED,
    logging_steps=50,
    save_total_limit=1,
    report_to='none',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)


# train
print('training...')
trainer.train()


# save best model
print('saving best model...')
trainer.save_model(str(MODEL_DIR/'best_model'))


# save results
val_results = trainer.evaluate(val_ds)
print('val results:', val_results)

results = {
    'model_name': args.model_name,
    'num_labels': NUM_LABELS,
    'val': val_results,
}
with open(MODEL_DIR/'results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'done.')