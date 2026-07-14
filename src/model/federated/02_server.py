import json
from pathlib import Path
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    hamming_loss,
    accuracy_score,
)
import torch
import numpy as np
from datasets import load_from_disk
from torch.utils.data import DataLoader


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SPL_DIR = BASE_DIR/'data'/'split'
PROC_DIR = BASE_DIR/'data'/'processed'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# load metadata
with open(PROC_DIR/'icd_category_meta.json') as f:
    meta = json.load(f)

NUM_LABELS = meta['num_labels']


# load parameters, run inference, return loss and metrics
def _run_eval(model, parameters, loader):
    keys = list(model.state_dict().keys())
    state_dict = dict(zip(keys, [torch.tensor(p) for p in parameters]))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    loss_fn = torch.nn.BCEWithLogitsLoss()
    all_probs = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].float().to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = (all_probs >= 0.5).astype(int)

    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    precision_micro = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    hamming = hamming_loss(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(loader)

    try:
        auc = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')
    except ValueError:
        auc = 0.0

    return avg_loss, {
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


# returns closure that evaluates global model on val set each round
def get_evaluate_fn(model_name, model_slug):
    val_ds = load_from_disk(str(SPL_DIR/model_slug/'val'))
    val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_loader = DataLoader(val_ds, batch_size=32)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        problem_type='multi_label_classification',
    ).to(DEVICE)

    def evaluate_fn(server_round, parameters, config):
        avg_loss, metrics = _run_eval(model, parameters, val_loader)
        print(f'[val] round {server_round} | loss={avg_loss:.4f} | f1_macro={metrics["f1_macro"]:.4f} | auc={metrics["auc"]:.4f}')
        return avg_loss, metrics

    return evaluate_fn


# returns closure that evaluates best model on test set after training
def get_test_evaluate_fn(model_name, model_slug):
    test_ds = load_from_disk(str(SPL_DIR/model_slug/'test'))
    test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_loader = DataLoader(test_ds, batch_size=32)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        problem_type='multi_label_classification',
    ).to(DEVICE)

    def test_evaluate_fn(parameters):
        avg_loss, metrics = _run_eval(model, parameters, test_loader)
        print(f'[test] loss={avg_loss:.4f} | f1_macro={metrics["f1_macro"]:.4f} | auc={metrics["auc"]:.4f}')
        return avg_loss, metrics

    return test_evaluate_fn