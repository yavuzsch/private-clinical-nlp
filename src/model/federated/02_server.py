import json
from pathlib import Path
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import f1_score, roc_auc_score
import torch
import flwr as fl
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


def get_evaluate_fn(model_name, model_slug):
    test_ds = load_from_disk(str(SPL_DIR/model_slug/'test'))
    test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_loader = DataLoader(test_ds, batch_size=32)

    def evaluate_fn(server_round, parameters, config):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=NUM_LABELS,
            problem_type='multi_label_classification',
        ).to(DEVICE)

        # set parameters
        keys = list(model.state_dict().keys())
        state_dict = dict(zip(keys, [torch.tensor(p) for p in parameters]))
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        loss_fn = torch.nn.BCEWithLogitsLoss()
        all_probs = []
        all_labels = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
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
        avg_loss = total_loss / len(test_loader)

        try:
            auc = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')
        except ValueError:
            auc = 0.0

        print(f'round {server_round} | loss={avg_loss:.4f} | f1_macro={f1_macro:.4f} | auc={auc:.4f}')

        return avg_loss, {'f1_macro': f1_macro, 'auc': auc}

    return evaluate_fn


def get_strategy(model_name, model_slug):
    evaluate_fn = get_evaluate_fn(model_name, model_slug)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=10,
        min_available_clients=10,
        evaluate_fn=evaluate_fn,
    )

    return strategy