import os
from datetime import date
from pathlib import Path
from src.utils.main_utils import read_yaml


PIPELINE_NAME: str = " "
ARTIFACT_DIR: str = "artifacts"

# Data Ingestion
DATA_INGESTION_ROOT_DIR: str = "data_ingestion"
DATA_INGESTION_SOURCE_URL: str = "https://drive.google.com/file/d/1dv0XttR6O_GXKVgcXifZHopX_nqNGR39/view?usp=sharing"
DATA_INGESTION_LOCAL_DATA_FILE: str = "data.zip"
DATA_INGESTION_UNZIP_DIR: str = "data_ingestion"

# Prepare Base Model
PREPARE_BASE_MODEL_ROOT_DIR: str = "prepare_base_model"
PREPARE_BASE_MODEL_BASE_MODEL_PATH: str = "base_model.h5"
PREPARE_BASE_MODEL_UPDATED_BASE_MODEL_PATH: str = "base_model_updated.h5"


# Model Trainer
TRAINING_ROOT_DIR: str = "training"
TRAINED_MODEL_PATH: str = "model.h5"
TRAINER_MODEL_PARAMS: str = os.path.join("config", "params.yaml")


# Model Evaluation
MLFLOW_URI: str = "https://dagshub.com/i-atul/Ship-Classifier.mlflow"
DAGSHUB_REPO_OWNER: str = "i-atul"
DAGSHUB_REPO_NAME: str = "Ship-Classifier"


# App Deployment
APP_HOST = "0.0.0.0"
APP_PORT = 5000
DEBUG = True
