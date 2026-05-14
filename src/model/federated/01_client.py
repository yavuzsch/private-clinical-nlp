import json
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import f1_score
import torch
import flwr as fl
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SPL_DIR = BASE_DIR/'data'/'split'
PROC_DIR = BASE_DIR/'data'/'processed'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# load metadata
with open(PROC_DIR/'icd_category_meta.json') as f:
    meta = json.load(f)

NUM_LABELS = meta['num_labels']


def get_parameters(model):
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model, parameters):
    keys = list(model.state_dict().keys())
    state_dict = dict(zip(keys, [torch.tensor(p) for p in parameters]))
    model.load_state_dict(state_dict, strict=True)


class ClinicalClient(fl.client.NumPyClient):
    def __init__(self, hospital_id, model_name, model_slug, learning_rate, batch_size, local_epochs):
        self.hospital_id = hospital_id
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs

        # load datasets
        hosp_dir = SPL_DIR/f'hospital_{hospital_id}'
        train_ds = load_from_disk(str(hosp_dir/model_slug/'train'))
        val_ds = load_from_disk(str(SPL_DIR/model_slug/'val'))

        train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        self.train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size * 2)

        self.train_size = len(train_ds)
        self.val_size = len(val_ds)

        # load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=NUM_LABELS,
            problem_type='multi_label_classification',
        ).to(DEVICE)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(self.local_epochs):
            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].float().to(DEVICE)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                optimizer.step()

        return get_parameters(self.model), self.train_size, {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)

        loss_fn = torch.nn.BCEWithLogitsLoss()
        self.model.eval()

        all_probs = []
        all_labels = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].float().to(DEVICE)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                total_loss += loss.item()

                probs = torch.sigmoid(outputs.logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = (all_probs >= 0.5).astype(int)

        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        avg_loss = total_loss / len(self.val_loader)

        return avg_loss, self.val_size, {'f1_macro': f1_macro}


def get_client_fn(model_name, learning_rate, batch_size, local_epochs):
    model_slug = model_name.replace('/', '_')
    def client_fn(context):
        hospital_id = int(context.node_id) % 10
        return ClinicalClient(
            hospital_id=hospital_id,
            model_name=model_name,
            model_slug=model_slug,
            learning_rate=learning_rate,
            batch_size=batch_size,
            local_epochs=local_epochs,
        ).to_client()
    return client_fn