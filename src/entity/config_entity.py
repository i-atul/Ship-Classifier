import os
from src.constants import *
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from src.utils.main_utils import read_yaml

params = read_yaml(Path("params.yaml"))

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR)
   

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_ROOT_DIR)
    source_URL: str = DATA_INGESTION_SOURCE_URL
    local_data_file: str = os.path.join(root_dir, DATA_INGESTION_LOCAL_DATA_FILE)
    unzip_dir: str = os.path.join(root_dir, DATA_INGESTION_UNZIP_DIR)

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path = Path(os.path.join(training_pipeline_config.artifact_dir, PREPARE_BASE_MODEL_ROOT_DIR))
    base_model_path: Path = root_dir / PREPARE_BASE_MODEL_BASE_MODEL_PATH
    updated_base_model_path: Path = root_dir / PREPARE_BASE_MODEL_UPDATED_BASE_MODEL_PATH
    params_image_size: list = params.IMAGE_SIZE
    params_learning_rate: float = params.LEARNING_RATE
    params_include_top: bool = params.INCLUDE_TOP
    params_weights: str = params.WEIGHTS
    params_classes: int = params.CLASSES

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path = Path(os.path.join(training_pipeline_config.artifact_dir, TRAINING_ROOT_DIR))
    trained_model_path: Path = root_dir / TRAINED_MODEL_PATH
    updated_base_model_path: Path = Path("artifacts/prepare_base_model/base_model.h5") 
    training_data: Path = Path(training_pipeline_config.artifact_dir) / DATA_INGESTION_ROOT_DIR / DATA_INGESTION_UNZIP_DIR
    params_epochs: int = params.EPOCHS
    params_batch_size: int = params.BATCH_SIZE
    params_is_augmentation: bool = params.AUGMENTATION
    params_image_size: list = params.IMAGE_SIZE
    params_classes: int = params.CLASSES
    params_learning_rate: float = params.LEARNING_RATE

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path = Path(os.path.join(training_pipeline_config.artifact_dir, TRAINING_ROOT_DIR, TRAINED_MODEL_PATH))
    training_data: Path = Path(os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_ROOT_DIR, DATA_INGESTION_UNZIP_DIR))
    all_params: dict = params
    mlflow_uri: str = MLFLOW_URI
    dagshub_repo_owner: str = DAGSHUB_REPO_OWNER
    dagshub_repo_name: str = DAGSHUB_REPO_NAME
    params_image_size: list = params.IMAGE_SIZE
    params_batch_size: int = params.BATCH_SIZE
    params_classes: int = params.CLASSES

@dataclass(frozen=True)
class DeploymentConfig:
    app_host : str = APP_HOST
    app_port: int = APP_PORT
    debug: bool = DEBUG
    