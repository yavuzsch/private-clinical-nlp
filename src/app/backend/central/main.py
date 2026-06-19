import json
import httpx
import torch
from contextlib import asynccontextmanager
from transformers import AutoModelForSequenceClassification
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from shared.config import (
    MODEL_SLUG, NUM_LABELS, MODELS_DIR,
    EPSILON, DELTA,
)
from shared.models import (
    RegisterRequest, RegisterResponse,
    HospitalInfo, SystemStatusResponse,
    AggregateResponse, GlobalModelResponse,
    MetricsResponse, WeightsPayload,
)


WEIGHTS_PATH = '/tmp/global_weights.pt'

# state
hospitals = {}  # hospital_id → HospitalInfo
total_rounds = 0


def load_global_model():
    """LOAD THE PRE-TRAINED FEDERATED DP MODEL AS STARTING POINT."""
    model_path = MODELS_DIR/'federated_dp'/MODEL_SLUG/f'epsilon_{EPSILON}'/'best_model'
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        num_labels=NUM_LABELS,
        problem_type='multi_label_classification',
        torch_dtype=torch.float16,
    )
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, WEIGHTS_PATH)

    # release memory
    del model
    del state_dict
    torch.cuda.empty_cache()
    print(f'central: global model loaded from {model_path}')


def read_weights():
    return torch.load(WEIGHTS_PATH, map_location='cpu')


def write_weights(weights: dict):
    torch.save(weights, WEIGHTS_PATH)


def load_metrics():
    """LOAD TEST METRICS FOR ALL MODELS."""
    metrics = []

    # centralized
    centralized_path = MODELS_DIR/'centralized'/MODEL_SLUG/'test_results.json'
    if centralized_path.exists():
        with open(centralized_path) as f:
            data = json.load(f)
        metrics.append(MetricsResponse(
            model_name='Centralized',
            epsilon=None,
            f1_macro=data['test']['f1_macro'],
            f1_micro=data['test']['f1_micro'],
            auc=data['test']['auc'],
            precision_macro=data['test']['precision_macro'],
            recall_macro=data['test']['recall_macro'],
            hamming_loss=data['test']['hamming_loss'],
        ))

    # federated
    federated_path = MODELS_DIR/'federated'/MODEL_SLUG/'test_results.json'
    if federated_path.exists():
        with open(federated_path) as f:
            data = json.load(f)
        metrics.append(MetricsResponse(
            model_name='Federated',
            epsilon=None,
            f1_macro=data['test']['test_f1_macro'],
            f1_micro=data['test']['test_f1_micro'],
            auc=data['test']['test_auc'],
            precision_macro=data['test']['test_precision_macro'],
            recall_macro=data['test']['test_recall_macro'],
            hamming_loss=data['test']['test_hamming_loss'],
        ))

    # federated dp
    for eps in [1.0, 3.0, 8.0]:
        dp_path = MODELS_DIR/'federated_dp'/MODEL_SLUG/f'epsilon_{eps}'/'test_results.json'
        if dp_path.exists():
            with open(dp_path) as f:
                data = json.load(f)
            metrics.append(MetricsResponse(
                model_name=f'Federated DP ε={eps}',
                epsilon=eps,
                f1_macro=data['test']['test_f1_macro'],
                f1_micro=data['test']['test_f1_micro'],
                auc=data['test']['test_auc'],
                precision_macro=data['test']['test_precision_macro'],
                recall_macro=data['test']['test_recall_macro'],
                hamming_loss=data['test']['test_hamming_loss'],
            ))

    return metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_global_model()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/hospitals/register', response_model=RegisterResponse)
async def register_hospital(req: RegisterRequest):
    """REGISTER A NEW HOSPITAL."""
    hospitals[req.hospital_id] = HospitalInfo(
        hospital_id=req.hospital_id,
        port=req.port,
        status='active',
        budget_remaining=EPSILON,
        notes_collected=0,
        rounds_completed=0,
    )
    print(f'central: hospital_{req.hospital_id} registered on port {req.port}')
    return RegisterResponse(
        hospital_id=req.hospital_id,
        status='active',
        epsilon=EPSILON,
        delta=DELTA,
        message=f'hospital_{req.hospital_id} registered successfully',
    )


@app.get('/hospitals', response_model=list[HospitalInfo])
async def list_hospitals():
    """LIST ALL HOSPITALS."""
    return list(hospitals.values())


@app.delete('/hospitals/{hospital_id}')
async def remove_hospital(hospital_id: int):
    """REMOVE A HOSPITAL FROM THE REGISTRY."""
    if hospital_id not in hospitals:
        raise HTTPException(status_code=404, detail='hospital not found')
    del hospitals[hospital_id]
    print(f'central: hospital_{hospital_id} removed')
    return {'message': f'hospital_{hospital_id} removed'}


@app.post('/aggregate', response_model=AggregateResponse)
async def aggregate(payload: WeightsPayload):
    """RECEIVE WEIGHTS FROM A HOSPITAL AND RUN FEDAVG."""
    global total_rounds

    if payload.hospital_id not in hospitals:
        raise HTTPException(status_code=404, detail='hospital not found')

    # update hospital info
    hospitals[payload.hospital_id].rounds_completed = payload.rounds_completed
    hospitals[payload.hospital_id].budget_remaining = payload.budget_remaining
    if payload.budget_remaining <= 0:
        hospitals[payload.hospital_id].status = 'frozen'

    # fedavg — average incoming weights with current global weights
    global_weights = read_weights()
    incoming = {k: torch.tensor(v, dtype=torch.float16) for k, v in payload.weights.items()}
    active_count = sum(1 for h in hospitals.values() if h.status == 'active')

    if active_count > 0:
        alpha = 1.0 / active_count
        new_weights = {}
        for k in global_weights:
            if k in incoming:
                new_weights[k] = ((1 - alpha) * global_weights[k] + alpha * incoming[k]).to(torch.float16)
            else:
                new_weights[k] = global_weights[k]
        write_weights(new_weights)
        del new_weights

    del global_weights
    del incoming

    total_rounds += 1
    print(f'central: fedavg complete — round {total_rounds}, hospital_{payload.hospital_id}')

    active_ids = [h.hospital_id for h in hospitals.values() if h.status == 'active']
    await distribute_global_model(active_ids)

    return AggregateResponse(
        rounds_completed=total_rounds,
        hospitals_included=active_ids,
        status='updated',
    )


async def distribute_global_model(hospital_ids: list[int]):
    """SEND UPDATED GLOBAL MODEL TO ALL ACTIVE HOSPITALS."""
    global_weights = read_weights()
    weights_serialized = {k: v.tolist() for k, v in global_weights.items()}
    del global_weights

    async with httpx.AsyncClient() as client:
        for h_id in hospital_ids:
            if h_id not in hospitals:
                continue
            port = hospitals[h_id].port
            try:
                await client.post(
                    f'http://hospital_{h_id}:{port}/update-model',
                    json={'weights': weights_serialized},
                    timeout=60,
                )
                print(f'central: model distributed to hospital_{h_id}')
            except Exception as e:
                print(f'central: failed to distribute to hospital_{h_id} — {e}')


@app.get('/global-model', response_model=GlobalModelResponse)
async def get_global_model():
    """GET CURRENT GLOBAL MODEL METADATA."""
    return GlobalModelResponse(
        rounds_completed=total_rounds,
        epsilon=EPSILON,
        delta=DELTA,
    )


@app.get('/global-weights')
async def get_global_weights():
    """GET CURRENT GLOBAL MODEL WEIGHTS — USED BY NEW HOSPITALS ON REGISTRATION."""
    global_weights = read_weights()
    result = {'weights': {k: v.tolist() for k, v in global_weights.items()}}
    del global_weights
    return result


@app.get('/status', response_model=SystemStatusResponse)
async def status():
    """GET SYSTEM STATUS."""
    active = sum(1 for h in hospitals.values() if h.status == 'active')
    frozen = sum(1 for h in hospitals.values() if h.status == 'frozen')
    return SystemStatusResponse(
        active_hospitals=active,
        frozen_hospitals=frozen,
        total_rounds=total_rounds,
        epsilon=EPSILON,
        delta=DELTA,
    )


@app.get('/models/metrics', response_model=list[MetricsResponse])
async def get_metrics():
    """GET TEST METRICS FOR ALL MODELS."""
    return load_metrics()