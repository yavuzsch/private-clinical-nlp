import json
import os
import csv
import httpx
import torch
from contextlib import asynccontextmanager
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from opacus import PrivacyEngine
from fastapi import FastAPI, HTTPException, BackgroundTasks

from shared.config import (
    MODEL_NAME, MODEL_SLUG, NUM_LABELS, MODELS_DIR, PROC_DIR, SPLIT_DIR,
    LOCAL_EPOCHS, BATCH_SIZE, LEARNING_RATE, MAX_GRAD_NORM, EPSILON, DELTA,
    NOTES_PER_ROUND, CENTRAL_PORT, HOSPITAL_BASE_PORT,
)
from shared.models import (
    NoteRequest, NoteResponse,
    PredictRequest, PredictResponse,
    TrainResponse, BudgetResponse,
    HospitalStatusResponse, UpdateModelRequest,
)


HOSPITAL_ID = int(os.getenv('HOSPITAL_ID', '0'))
CENTRAL_URL = f'http://central:{CENTRAL_PORT}'

# state
model = None
tokenizer = None
categories = []
notes_buffer = []  # list of {'text': str, 'predictions': list[int]}
rounds_completed = 0
budget_remaining = EPSILON
is_frozen = False


def categories_to_predictions(icd_str: str, categories: list[str]) -> list[int]:
    """CONVERT PIPE-SEPARATED ICD CATEGORIES STRING TO BINARY PREDICTION VECTOR."""
    active = set(icd_str.split('|'))
    return [1 if cat in active else 0 for cat in categories]


def load_model():
    """LOAD THE HOSPITAL'S LOCAL MODEL AND SEED BUFFER WITH LOCAL DATA."""
    global model, tokenizer, categories, notes_buffer

    # load categories
    with open(PROC_DIR/'icd_category_meta.json') as f:
        meta = json.load(f)
    categories = meta['categories']

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load model — use pre-trained federated dp model as starting point
    model_path = MODELS_DIR/'federated_dp'/MODEL_SLUG/f'epsilon_{EPSILON}'/'best_model'
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        num_labels=NUM_LABELS,
        problem_type='multi_label_classification',
    )
    model.eval()
    print(f'hospital_{HOSPITAL_ID}: model loaded from {model_path}')

    # load initial local data into buffer
    local_csv = SPLIT_DIR/f'hospital_{HOSPITAL_ID}'/'train.csv'
    if local_csv.exists():
        with open(local_csv, newline='') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= NOTES_PER_ROUND:
                    break
                notes_buffer.append({
                    'text': row['TEXT'],
                    'predictions': categories_to_predictions(row['ICD_CATEGORIES'], categories),
                })
        print(f'hospital_{HOSPITAL_ID}: {len(notes_buffer)} notes loaded from local data')


async def register_with_central():
    """REGISTER WITH CENTRAL SERVER."""
    port = HOSPITAL_BASE_PORT + HOSPITAL_ID
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f'{CENTRAL_URL}/hospitals/register',
                json={'hospital_id': HOSPITAL_ID, 'port': port},
                timeout=30,
            )
            resp.raise_for_status()
            print(f'hospital_{HOSPITAL_ID}: registered with central server')

        except Exception as e:
            print(f'hospital_{HOSPITAL_ID}: failed to register — {e}')


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    await register_with_central()
    yield


app = FastAPI(lifespan=lifespan)


@app.get('/status', response_model=HospitalStatusResponse)
async def status():
    """GET HOSPITAL STATUS."""
    return HospitalStatusResponse(
        hospital_id=HOSPITAL_ID,
        status='frozen' if is_frozen else 'active',
        budget_remaining=budget_remaining,
        notes_collected=len(notes_buffer),
        rounds_completed=rounds_completed,
    )


@app.get('/budget', response_model=BudgetResponse)
async def budget():
    """GET REMAINING PRIVACY BUDGET."""
    return BudgetResponse(
        hospital_id=HOSPITAL_ID,
        epsilon=EPSILON,
        budget_remaining=budget_remaining,
        status='frozen' if is_frozen else 'active',
    )


@app.post('/predict', response_model=PredictResponse)
async def predict(req: PredictRequest):
    """RUN LOCAL INFERENCE — NOTE NEVER LEAVES THIS HOSPITAL."""
    if model is None:
        raise HTTPException(status_code=503, detail='model not loaded')

    inputs = tokenizer(
        req.text,
        return_tensors='pt',
        max_length=256,
        truncation=True,
        padding=True,
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.sigmoid(logits).squeeze().tolist()
    preds = [1 if p >= 0.5 else 0 for p in probs]

    predicted_categories = [categories[i] for i, p in enumerate(preds) if p == 1]

    return PredictResponse(
        hospital_id=HOSPITAL_ID,
        predictions=preds,
        probabilities=probs,
        categories=predicted_categories,
    )


@app.post('/notes', response_model=NoteResponse)
async def add_note(req: NoteRequest, background_tasks: BackgroundTasks):
    """ADD NOTE TO LOCAL BUFFER — TRIGGERS TRAINING WHEN BUFFER IS FULL."""
    global notes_buffer

    # store text and confirmed predictions together
    notes_buffer.append({'text': req.text, 'predictions': req.predictions})
    notes_until_round = max(0, NOTES_PER_ROUND - len(notes_buffer))

    # trigger training in background when buffer is full
    if len(notes_buffer) >= NOTES_PER_ROUND and not is_frozen:
        background_tasks.add_task(run_training)

    return NoteResponse(
        hospital_id=HOSPITAL_ID,
        notes_collected=len(notes_buffer),
        notes_until_round=notes_until_round,
        message='note added' if notes_until_round > 0 else 'round triggered',
    )


async def run_training():
    """RUN LOCAL DP TRAINING AND SEND WEIGHTS TO CENTRAL SERVER."""
    global model, notes_buffer, rounds_completed, budget_remaining, is_frozen

    if is_frozen:
        print(f'hospital_{HOSPITAL_ID}: budget exhausted, skipping training')
        return

    if budget_remaining <= 0:
        is_frozen = True
        print(f'hospital_{HOSPITAL_ID}: budget exhausted, model frozen')
        return

    print(f'hospital_{HOSPITAL_ID}: starting local dp training...')

    # prepare texts and labels from buffer
    texts = [n['text'] for n in notes_buffer]
    labels = torch.tensor([n['predictions'] for n in notes_buffer], dtype=torch.float)

    encodings = tokenizer(
        texts,
        max_length=256,
        truncation=True,
        padding=True,
        return_tensors='pt',
    )

    dataset = [
        {k: v[i] for k, v in encodings.items()} | {'labels': labels[i]}
        for i in range(len(texts))
    ]

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # dp training
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    privacy_engine = PrivacyEngine()
    model_dp, optimizer_dp, loader_dp = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        epochs=LOCAL_EPOCHS,
        target_epsilon=min(budget_remaining, EPSILON),
        target_delta=DELTA,
        max_grad_norm=MAX_GRAD_NORM,
    )

    for epoch in range(LOCAL_EPOCHS):
        for batch in loader_dp:
            optimizer_dp.zero_grad()
            outputs = model_dp(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
            )
            outputs.loss.backward()
            optimizer_dp.step()

    # update budget
    epsilon_used = privacy_engine.get_epsilon(DELTA)
    budget_remaining = max(0, budget_remaining - epsilon_used)
    rounds_completed += 1

    if budget_remaining <= 0:
        is_frozen = True
        print(f'hospital_{HOSPITAL_ID}: budget exhausted after round {rounds_completed}')

    model.eval()

    # send weights to central
    weights = {k: v.cpu().tolist() for k, v in model_dp._module.state_dict().items()}

    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                f'{CENTRAL_URL}/aggregate',
                json={
                    'hospital_id': HOSPITAL_ID,
                    'weights': weights,
                    'rounds_completed': rounds_completed,
                    'budget_remaining': budget_remaining,
                },
                timeout=120,
            )
            print(f'hospital_{HOSPITAL_ID}: weights sent to central')
        except Exception as e:
            print(f'hospital_{HOSPITAL_ID}: failed to send weights — {e}')

    # clear buffer
    notes_buffer = []


@app.post('/train', response_model=TrainResponse)
async def train(background_tasks: BackgroundTasks):
    """MANUALLY TRIGGER LOCAL TRAINING."""
    background_tasks.add_task(run_training)
    return TrainResponse(
        hospital_id=HOSPITAL_ID,
        rounds_completed=rounds_completed,
        budget_remaining=budget_remaining,
        status='frozen' if is_frozen else 'training',
    )


@app.post('/update-model')
async def update_model(req: UpdateModelRequest):
    """RECEIVE UPDATED GLOBAL MODEL FROM CENTRAL SERVER."""
    global model

    if model is None:
        raise HTTPException(status_code=503, detail='model not loaded')

    # load new weights into model
    new_state_dict = {k: torch.tensor(v) for k, v in req.weights.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    print(f'hospital_{HOSPITAL_ID}: global model updated')
    return {'message': 'model updated successfully'}