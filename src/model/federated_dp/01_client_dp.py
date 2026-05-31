import json
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification
import torch
import flwr as fl
from opacus import PrivacyEngine


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SPL_DIR = BASE_DIR/'data'/'split'
PROC_DIR = BASE_DIR/'data'/'processed'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# load metadata
with open(PROC_DIR/'icd_category_meta.json') as f:
    meta = json.load(f)

NUM_LABELS = meta['num_labels']


# extract model weights as numpy arrays
def get_parameters(model):
    return [val.cpu().numpy() for val in model.state_dict().values()]


# load numpy arrays into model weights
def set_parameters(model, parameters):
    keys = list(model.state_dict().keys())
    state_dict = dict(zip(keys, [torch.tensor(p) for p in parameters]))
    model.load_state_dict(state_dict, strict=True)


class ClinicalClientDP(fl.client.NumPyClient):
    def __init__(self, hospital_id, model_name, model_slug, learning_rate, batch_size, local_epochs, epsilon, delta, max_grad_norm):
        self.hospital_id = hospital_id
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm

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
        self.model.train()

        # freeze embeddings for opacus compatibility
        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = False

        # setup optimizer and loss
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # apply privacy engine
        privacy_engine = PrivacyEngine()
        private_model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            epochs=self.local_epochs,
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            max_grad_norm=self.max_grad_norm,
        )

        private_model.train()
        for epoch in range(self.local_epochs):
            for batch in train_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].float().to(DEVICE)

                optimizer.zero_grad()
                outputs = private_model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                optimizer.step()

        return get_parameters(private_model._module), self.train_size, {}