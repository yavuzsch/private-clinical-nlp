import sys
import json
import argparse
import importlib
from pathlib import Path

import torch
import flwr as fl
from transformers import AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).resolve().parent))

client_module = importlib.import_module('01_client')
server_module = importlib.import_module('02_server')

get_client_fn = client_module.get_client_fn
get_strategy = server_module.get_strategy


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODEL_DIR = BASE_DIR/'models'/'federated'


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='emilyalsentzer/Bio_ClinicalBERT')
parser.add_argument('--num_rounds', type=int, default=10)
parser.add_argument('--local_epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--num_clients', type=int, default=10)
args = parser.parse_args()

MODEL_SLUG = args.model_name.replace('/', '_')
RUN_DIR = MODEL_DIR/MODEL_SLUG
RUN_DIR.mkdir(parents=True, exist_ok=True)

print(f'model : {args.model_name}')
print(f'num_rounds : {args.num_rounds}')
print(f'local_epochs : {args.local_epochs}')
print(f'num_clients : {args.num_clients}')


# load model for initial parameters
print('loading initial model...')
with open(BASE_DIR/'data'/'processed'/'icd_category_meta.json') as f:
    meta = json.load(f)

model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=meta['num_labels'],
    problem_type='multi_label_classification',
)
initial_parameters = fl.common.ndarrays_to_parameters(
    [val.cpu().numpy() for val in model.state_dict().values()]
)
del model


# client and strategy
client_fn = get_client_fn(
    model_name=args.model_name,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    local_epochs=args.local_epochs,
)

strategy = get_strategy(
    model_name=args.model_name,
    model_slug=MODEL_SLUG,
)
strategy.initial_parameters = initial_parameters


# run simulation
print('starting federated simulation...')
history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=args.num_clients,
    config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    strategy=strategy,
    client_resources={'num_cpus': 1, 'num_gpus': 0.5},
)


# save results
results = {
    'model_name': args.model_name,
    'num_rounds': args.num_rounds,
    'local_epochs': args.local_epochs,
    'num_clients': args.num_clients,
    'learning_rate': args.learning_rate,
    'batch_size': args.batch_size,
    'losses_distributed': history.losses_distributed,
    'metrics_distributed': history.metrics_distributed,
    'losses_centralized': history.losses_centralized,
    'metrics_centralized': history.metrics_centralized,
}
with open(RUN_DIR/'results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print('done.')