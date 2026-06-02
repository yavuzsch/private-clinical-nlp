import sys
import json
import argparse
import importlib
import random
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

sys.path.insert(0, str(Path(__file__).resolve().parent))

client_module = importlib.import_module('01_client')
server_module = importlib.import_module('02_server')

ClinicalClient = client_module.ClinicalClient
get_evaluate_fn = server_module.get_evaluate_fn
get_test_evaluate_fn = server_module.get_test_evaluate_fn


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


# load metadata
with open(BASE_DIR/'data'/'processed'/'icd_category_meta.json') as f:
    meta = json.load(f)

NUM_LABELS = meta['num_labels']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# initialize clients
print('initializing clients...')
clients = [
    ClinicalClient(
        hospital_id=i,
        model_name=args.model_name,
        model_slug=MODEL_SLUG,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
    )
    for i in range(args.num_clients)
]


# evaluate functions
evaluate_fn = get_evaluate_fn(model_name=args.model_name, model_slug=MODEL_SLUG)
test_evaluate_fn = get_test_evaluate_fn(model_name=args.model_name, model_slug=MODEL_SLUG)


# initialize global model
print('loading initial model...')
global_model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=NUM_LABELS,
    problem_type='multi_label_classification',
).to(DEVICE)
global_parameters = [val.cpu().numpy() for val in global_model.state_dict().values()]
del global_model


# federated simulation
print('starting federated simulation...')
history = []
best_f1 = 0.0
best_parameters = None

for round_num in range(1, args.num_rounds + 1):
    print(f'\nround {round_num}/{args.num_rounds}')

    # local training
    all_parameters = []
    all_sizes = []
    for client in clients:
        parameters, size, _ = client.fit(global_parameters, {})
        all_parameters.append(parameters)
        all_sizes.append(size)

    # fedavg
    total = sum(all_sizes)
    global_parameters = [
        np.sum([p[i] * s / total for p, s in zip(all_parameters, all_sizes)], axis=0)
        for i in range(len(global_parameters))
    ]

    # val evaluation after each round
    loss, metrics = evaluate_fn(round_num, global_parameters, {})

    history.append({
        'round': round_num,
        'loss': loss,
        'f1_macro': metrics['f1_macro'],
        'f1_micro': metrics['f1_micro'],
        'f1_weighted': metrics['f1_weighted'],
        'precision_macro': metrics['precision_macro'],
        'precision_micro': metrics['precision_micro'],
        'recall_macro': metrics['recall_macro'],
        'recall_micro': metrics['recall_micro'],
        'hamming_loss': metrics['hamming_loss'],
        'accuracy': metrics['accuracy'],
        'auc': metrics['auc'],
    })

    if metrics['f1_macro'] > best_f1:
        best_f1 = metrics['f1_macro']
        best_parameters = [p.copy() for p in global_parameters]
        print(f'new best f1_macro: {best_f1:.4f}')


# test evaluation
print('running test evaluation...')
test_loss, test_metrics = test_evaluate_fn(best_parameters)


# save best model
print('saving best model...')
best_model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=NUM_LABELS,
    problem_type='multi_label_classification',
)
keys = list(best_model.state_dict().keys())
state_dict = dict(zip(keys, [torch.tensor(p) for p in best_parameters]))
best_model.load_state_dict(state_dict, strict=True)
best_model.save_pretrained(str(RUN_DIR/'best_model'))
print(f'best model saved. f1_macro={best_f1:.4f}')


# save results
results = {
    'model_name': args.model_name,
    'num_rounds': args.num_rounds,
    'local_epochs': args.local_epochs,
    'num_clients': args.num_clients,
    'learning_rate': args.learning_rate,
    'batch_size': args.batch_size,
    'best_f1_macro': best_f1,
    'history': history,
}
with open(RUN_DIR/'results.json', 'w') as f:
    json.dump(results, f, indent=2)

test_results = {
    'model_name': args.model_name,
    'num_labels': NUM_LABELS,
    'test': {
        'test_loss': test_loss,
        'test_f1_macro': test_metrics['f1_macro'],
        'test_f1_micro': test_metrics['f1_micro'],
        'test_f1_weighted': test_metrics['f1_weighted'],
        'test_precision_macro': test_metrics['precision_macro'],
        'test_precision_micro': test_metrics['precision_micro'],
        'test_recall_macro': test_metrics['recall_macro'],
        'test_recall_micro': test_metrics['recall_micro'],
        'test_hamming_loss': test_metrics['hamming_loss'],
        'test_accuracy': test_metrics['accuracy'],
        'test_auc': test_metrics['auc'],
    }
}
with open(RUN_DIR/'test_results.json', 'w') as f:
    json.dump(test_results, f, indent=2)

print('done.')