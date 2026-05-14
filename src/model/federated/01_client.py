import json
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification
import torch
import flwr as fl


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

        # load train dataset
        hosp_dir = SPL_DIR/f'hospital_{hospital_id}'
        train_ds = load_from_disk(str(hosp_dir/model_slug/'train'))
        train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        self.train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.train_size = len(train_ds)

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