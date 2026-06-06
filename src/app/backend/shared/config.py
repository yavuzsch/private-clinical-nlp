import os
from pathlib import Path


BASE_DIR = Path(os.getenv('BASE_DIR', str(Path(__file__).resolve().parent.parent.parent.parent)))
MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
MODEL_SLUG = MODEL_NAME.replace('/', '_')
NUM_LABELS = 19

# paths
MODELS_DIR = BASE_DIR/'models'
DATA_DIR = BASE_DIR/'data'
SPLIT_DIR = DATA_DIR/'split'
PROC_DIR = DATA_DIR/'processed'
NOTES_DIR = BASE_DIR/'notes'

# federated
NUM_HOSPITALS = 10
NOTES_PER_ROUND = 5

# training
LOCAL_EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_GRAD_NORM = 3.0

# privacy
EPSILON = float(os.getenv('EPSILON', '8.0'))
DELTA = 1e-5

# auth
HOSPITAL_PASSWORD = os.getenv('HOSPITAL_PASSWORD', 'hospital')
JWT_SECRET = os.getenv('JWT_SECRET', 'secret')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRE_MINUTES = 60 * 24  # 24 hours

# ports
CENTRAL_PORT = 8000
HOSPITAL_BASE_PORT = 8100