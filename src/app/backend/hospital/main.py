import json
import os
import csv
import uuid
import httpx
import torch
import asyncio
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from opacus import PrivacyEngine
from jose import JWTError, jwt
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

from shared.config import (
    MODEL_NAME, MODEL_SLUG, NUM_LABELS, MODELS_DIR, PROC_DIR, SPLIT_DIR, NOTES_DIR,
    LOCAL_EPOCHS, BATCH_SIZE, LEARNING_RATE, MAX_GRAD_NORM, EPSILON, DELTA,
    NOTES_PER_ROUND, CENTRAL_PORT, HOSPITAL_BASE_PORT,
    HOSPITAL_PASSWORD, JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRE_MINUTES,
)
from shared.models import (
    NoteRequest, NoteResponse, NoteRecord, NoteUpdateRequest,
    PredictRequest, PredictResponse,
    TrainResponse, BudgetResponse,
    HospitalStatusResponse, UpdateModelRequest,
    LoginRequest, LoginResponse,
)


HOSPITAL_ID = int(os.getenv('HOSPITAL_ID', '0'))
CENTRAL_URL = f'http://central:{CENTRAL_PORT}'
NOTES_FILE = NOTES_DIR/f'hospital_{HOSPITAL_ID}.json'

security = HTTPBearer()

# state
model = None
tokenizer = None
categories = []
rounds_completed = 0
budget_remaining = EPSILON
is_frozen = False


def load_notes() -> list[dict]:
    """LOAD NOTES FROM JSON FILE."""
    if not NOTES_FILE.exists():
        return []
    with open(NOTES_FILE) as f:
        return json.load(f)


def save_notes(notes: list[dict]):
    """SAVE NOTES TO JSON FILE."""
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    with open(NOTES_FILE, 'w') as f:
        json.dump(notes, f, indent=2)


def get_buffer_notes() -> list[dict]:
    """GET NOTES NOT YET USED IN TRAINING."""
    return [n for n in load_notes() if not n['used_in_training']]


def create_token(hospital_id: int) -> str:
    """CREATE JWT TOKEN FOR HOSPITAL."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    return jwt.encode(
        {'hospital_id': hospital_id, 'exp': expire},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> int:
    """VERIFY JWT TOKEN AND RETURN HOSPITAL ID."""
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        hospital_id = payload.get('hospital_id')
        if hospital_id != HOSPITAL_ID:
            raise HTTPException(status_code=403, detail='token does not match this hospital')
        return hospital_id
    except JWTError:
        raise HTTPException(status_code=401, detail='invalid token')


def categories_to_predictions(icd_str: str, categories: list[str]) -> list[int]:
    """CONVERT PIPE-SEPARATED ICD CATEGORIES STRING TO BINARY PREDICTION VECTOR."""
    active = set(icd_str.split('|'))
    return [1 if cat in active else 0 for cat in categories]


def load_categories():
    """LOAD CATEGORY LIST, TOKENIZER, AND SEED NOTES — RUNS EAGERLY AT STARTUP."""
    global tokenizer, categories

    # load categories
    with open(PROC_DIR/'icd_category_meta.json') as f:
        meta = json.load(f)
    categories = meta['categories']

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR/'tokenizer'))

    # seed notes file with initial local data if empty
    if not NOTES_FILE.exists():
        NOTES_DIR.mkdir(parents=True, exist_ok=True)
        notes = []
        local_csv = SPLIT_DIR/f'hospital_{HOSPITAL_ID}'/'train.csv'
        if local_csv.exists():
            with open(local_csv, newline='') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= NOTES_PER_ROUND:
                        break
                    preds = categories_to_predictions(row['ICD_CATEGORIES'], categories)
                    notes.append({
                        'id': str(uuid.uuid4()),
                        'text': row['TEXT'],
                        'predictions': preds,
                        'categories': [categories[j] for j, p in enumerate(preds) if p == 1],
                        'used_in_training': False,
                        'created_at': datetime.now(timezone.utc).isoformat(),
                    })
            save_notes(notes)
        print(f'hospital_{HOSPITAL_ID}: {len(notes)} notes seeded from local data')


def ensure_model_loaded():
    """LAZILY LOAD THE HOSPITAL'S LOCAL MODEL ON FIRST USE (PREDICT / TRAIN / UPDATE-MODEL)."""
    global model
    if model is not None:
        return

    model_path = MODELS_DIR/'federated_dp'/MODEL_SLUG/f'epsilon_{EPSILON}'/'best_model'
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), num_labels=NUM_LABELS,
        problem_type='multi_label_classification', low_cpu_mem_usage=True,
    )
    model.eval()
    print(f'hospital_{HOSPITAL_ID}: model loaded from {model_path}')


async def register_with_central():
    """REGISTER WITH CENTRAL SERVER."""
    port = HOSPITAL_BASE_PORT + HOSPITAL_ID
    for attempt in range(5):
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f'{CENTRAL_URL}/hospitals/register',
                    json={'hospital_id': HOSPITAL_ID, 'port': port},
                    timeout=30,
                )
                resp.raise_for_status()
                print(f'hospital_{HOSPITAL_ID}: registered with central server')
                return
            except Exception as e:
                print(f'hospital_{HOSPITAL_ID}: register attempt {attempt+1} failed — {e}')
                await asyncio.sleep(3)
    print(f'hospital_{HOSPITAL_ID}: could not register after 5 attempts')


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_categories()
    asyncio.create_task(delayed_register())
    yield

async def delayed_register():
    await asyncio.sleep(10)
    await register_with_central()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/categories')
async def get_categories():
    """GET ICD CATEGORY LIST IN MODEL ORDER."""
    return {'categories': categories}


@app.post('/auth/login', response_model=LoginResponse)
async def login(req: LoginRequest):
    """LOGIN AND RECEIVE JWT TOKEN."""
    if req.hospital_id != HOSPITAL_ID or req.password != HOSPITAL_PASSWORD:
        raise HTTPException(status_code=401, detail='invalid credentials')
    token = create_token(HOSPITAL_ID)
    return LoginResponse(access_token=token, token_type='bearer', hospital_id=HOSPITAL_ID)


@app.get('/status', response_model=HospitalStatusResponse)
async def status(_: int = Depends(verify_token)):
    """GET HOSPITAL STATUS."""
    buffer_count = len(get_buffer_notes())
    return HospitalStatusResponse(
        hospital_id=HOSPITAL_ID,
        status='frozen' if is_frozen else 'active',
        budget_remaining=budget_remaining,
        notes_collected=buffer_count,
        rounds_completed=rounds_completed,
    )


@app.get('/budget', response_model=BudgetResponse)
async def budget(_: int = Depends(verify_token)):
    """GET REMAINING PRIVACY BUDGET."""
    return BudgetResponse(
        hospital_id=HOSPITAL_ID,
        epsilon=EPSILON,
        budget_remaining=budget_remaining,
        status='frozen' if is_frozen else 'active',
    )


@app.post('/predict', response_model=PredictResponse)
async def predict(req: PredictRequest, _: int = Depends(verify_token)):
    """RUN LOCAL INFERENCE — NOTE NEVER LEAVES THIS HOSPITAL."""
    ensure_model_loaded()

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


@app.get('/notes', response_model=list[NoteRecord])
async def get_notes(_: int = Depends(verify_token)):
    """GET ALL NOTES."""
    return load_notes()


@app.post('/notes', response_model=NoteResponse)
async def add_note(req: NoteRequest, background_tasks: BackgroundTasks, _: int = Depends(verify_token)):
    """ADD NOTE — TRIGGERS TRAINING WHEN BUFFER IS FULL."""
    notes = load_notes()

    # add new note
    preds = req.predictions
    note = {
        'id': str(uuid.uuid4()),
        'text': req.text,
        'predictions': preds,
        'categories': [categories[i] for i, p in enumerate(preds) if p == 1],
        'used_in_training': False,
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    notes.append(note)
    save_notes(notes)

    buffer_count = len(get_buffer_notes())
    notes_until_round = max(0, NOTES_PER_ROUND - buffer_count)

    # trigger training in background when buffer is full
    if buffer_count >= NOTES_PER_ROUND and not is_frozen:
        background_tasks.add_task(run_training)

    return NoteResponse(
        hospital_id=HOSPITAL_ID,
        notes_collected=buffer_count,
        notes_until_round=notes_until_round,
        message='note added' if notes_until_round > 0 else 'round triggered',
    )


@app.delete('/notes/{note_id}')
async def delete_note(note_id: str, _: int = Depends(verify_token)):
    """DELETE A NOTE."""
    notes = load_notes()
    updated = [n for n in notes if n['id'] != note_id]
    if len(updated) == len(notes):
        raise HTTPException(status_code=404, detail='note not found')
    save_notes(updated)
    return {'message': 'note deleted'}


@app.patch('/notes/{note_id}', response_model=NoteRecord)
async def update_note(note_id: str, req: NoteUpdateRequest, _: int = Depends(verify_token)):
    """UPDATE A NOTE."""
    notes = load_notes()
    for note in notes:
        if note['id'] == note_id:
            if req.text is not None:
                note['text'] = req.text
            if req.predictions is not None:
                note['predictions'] = req.predictions
                note['categories'] = [categories[i] for i, p in enumerate(req.predictions) if p == 1]
            save_notes(notes)
            return note
    raise HTTPException(status_code=404, detail='note not found')


async def run_training(force: bool = False):
    """RUN LOCAL DP TRAINING AND SEND WEIGHTS TO CENTRAL SERVER."""
    global model, rounds_completed, budget_remaining, is_frozen

    if is_frozen:
        print(f'hospital_{HOSPITAL_ID}: budget exhausted, skipping training')
        return

    if budget_remaining <= 0:
        is_frozen = True
        print(f'hospital_{HOSPITAL_ID}: budget exhausted, model frozen')
        return

    # get buffer notes
    buffer_notes = get_buffer_notes()

    if not force and len(buffer_notes) < NOTES_PER_ROUND:
        return

    if len(buffer_notes) == 0:
        print(f'hospital_{HOSPITAL_ID}: no pending notes, skipping training')
        return

    print(f'hospital_{HOSPITAL_ID}: starting local dp training...')

    # prepare texts and labels from buffer
    texts = [n['text'] for n in buffer_notes]
    labels = torch.tensor([n['predictions'] for n in buffer_notes], dtype=torch.float)

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
    ensure_model_loaded()
    model.train()

    # freeze embeddings and first 6 layers for opacus compatibility
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False

    for i in range(6):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

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

    # mark buffer notes as used in training
    notes = load_notes()
    buffer_ids = {n['id'] for n in buffer_notes}
    for note in notes:
        if note['id'] in buffer_ids:
            note['used_in_training'] = True
    save_notes(notes)

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


@app.post('/train', response_model=TrainResponse)
async def train(background_tasks: BackgroundTasks, _: int = Depends(verify_token)):
    """MANUALLY TRIGGER LOCAL TRAINING."""
    if is_frozen:
        return TrainResponse(
            hospital_id=HOSPITAL_ID,
            rounds_completed=rounds_completed,
            budget_remaining=budget_remaining,
            status='frozen',
        )

    if len(get_buffer_notes()) == 0:
        return TrainResponse(
            hospital_id=HOSPITAL_ID,
            rounds_completed=rounds_completed,
            budget_remaining=budget_remaining,
            status='no_pending_notes',
        )

    background_tasks.add_task(run_training, force=True)
    return TrainResponse(
        hospital_id=HOSPITAL_ID,
        rounds_completed=rounds_completed,
        budget_remaining=budget_remaining,
        status='training',
    )


@app.post('/update-model')
async def update_model(req: UpdateModelRequest):
    """RECEIVE UPDATED GLOBAL MODEL FROM CENTRAL SERVER."""
    global model
    ensure_model_loaded()

    # load new weights into model
    new_state_dict = {k: torch.tensor(v) for k, v in req.weights.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    print(f'hospital_{HOSPITAL_ID}: global model updated')
    return {'message': 'model updated successfully'}