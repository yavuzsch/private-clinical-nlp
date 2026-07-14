# App
Docker-based federated learning application for privacy-preserving ICD-9 category prediction from clinical notes.


## Requirements
- Docker
- Docker Compose


## Setup
Build the base image first:

```bash
docker build -f src/app/backend/Dockerfile.base -t clinical-nlp-base .
```

Copy `.env.example` to `.env` and configure if needed:

```bash
cp .env.example .env
```

Start all services:

```bash
docker compose up --build
```


## Usage
Open `http://localhost:3000` and log in with your hospital ID (0–9) and password.


## Services
| Service | Port |
|---------|------|
| Frontend | 3000 |
| Central server | 8000 |
| Hospital 0–9 | 8100–8109 |


## Structure
```
private-clinical-nlp/
├── docker-compose.yml
├── .env
├── notes/  # per-hospital note storage
└── src/
    └── app/
        ├── backend/
        │   ├── Dockerfile.base
        │   ├── central/  # fedavg aggregation server
        │   ├── hospital/  # local inference + dp training
        │   └── shared/  # config, pydantic models
        └── frontend/  # react + vite
```


## Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `EPSILON` | `8.0` | Differential privacy budget |
| `JWT_SECRET` | `secret` | JWT signing key |
| `HOSPITAL_N_PASSWORD` | `hospital` | Login password per hospital |