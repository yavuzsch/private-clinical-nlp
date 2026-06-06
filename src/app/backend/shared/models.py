from typing import Optional
from pydantic import BaseModel


class RegisterRequest(BaseModel):
    hospital_id: int
    port: int


class RegisterResponse(BaseModel):
    hospital_id: int
    status: str
    epsilon: float
    delta: float
    message: str


class HospitalInfo(BaseModel):
    hospital_id: int
    port: int
    status: str  # active, frozen
    budget_remaining: float
    notes_collected: int
    rounds_completed: int


class NoteRequest(BaseModel):
    text: str
    predictions: list[int]  # confirmed labels from user


class NoteRecord(BaseModel):
    id: str
    text: str
    predictions: list[int]
    categories: list[str]
    used_in_training: bool
    created_at: str


class NoteResponse(BaseModel):
    hospital_id: int
    notes_collected: int
    notes_until_round: int
    message: str


class NoteUpdateRequest(BaseModel):
    text: Optional[str] = None
    predictions: Optional[list[int]] = None


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    hospital_id: int
    predictions: list[int]
    probabilities: list[float]
    categories: list[str]


class WeightsPayload(BaseModel):
    hospital_id: int
    weights: dict[str, list]  # layer name → weight values
    rounds_completed: int
    budget_remaining: float


class TrainResponse(BaseModel):
    hospital_id: int
    rounds_completed: int
    budget_remaining: float
    status: str  # trained, frozen


class AggregateResponse(BaseModel):
    rounds_completed: int
    hospitals_included: list[int]
    status: str


class GlobalModelResponse(BaseModel):
    rounds_completed: int
    epsilon: float
    delta: float


class BudgetResponse(BaseModel):
    hospital_id: int
    epsilon: float
    budget_remaining: float
    status: str  # active, frozen


class HospitalStatusResponse(BaseModel):
    hospital_id: int
    status: str  # active, frozen
    budget_remaining: float
    notes_collected: int
    rounds_completed: int


class SystemStatusResponse(BaseModel):
    active_hospitals: int
    frozen_hospitals: int
    total_rounds: int
    epsilon: float
    delta: float


class MetricsResponse(BaseModel):
    model_name: str
    epsilon: Optional[float]
    f1_macro: float
    f1_micro: float
    auc: float
    precision_macro: float
    recall_macro: float
    hamming_loss: float


class UpdateModelRequest(BaseModel):
    weights: dict[str, list]


class LoginRequest(BaseModel):
    hospital_id: int
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str  # bearer
    hospital_id: int