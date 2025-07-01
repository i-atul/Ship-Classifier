import sys
from src.exception import my_exception as exception
from src.logger import my_logger

from src.components.data_ingestion import DataIngestion
from src.components.base_model import PrepareBaseModel
from src.components.model_trainer import Training 
from src.components.model_evaluation import Evaluation

from src.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig
)

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.base_model_config = PrepareBaseModelConfig()
        self.model_trainer_config = TrainingConfig()
        self.model_evaluation_config = EvaluationConfig()

    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion(config=self.data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()
            return True
        except Exception as e:
            raise exception(e, sys) from e

    def start_prepare_base_model(self):
        try:
            base_model = PrepareBaseModel(config=self.base_model_config)
            base_model.get_base_model()
            return True
        except Exception as e:
            raise exception(e, sys) from e


    def start_model_trainer(self):
        try:
            model_trainer = Training(config=self.model_trainer_config)
            model_trainer.get_base_model()  # Ensure base model is loaded and classifier is set
            model_trainer.train_valid_loader()  # Prepare data loaders
            model_trainer.train()  # Run training loop
            return True
        except Exception as e:
            raise exception(e, sys) from e
    

    def start_model_evaluation(self):
        try:
            model_evaluator = Evaluation(config=self.model_evaluation_config)
            model_evaluator.evaluate()
            model_evaluator.save_score()
            model_evaluator.log_to_mlflow()
            return True
        except Exception as e:
            raise exception(e, sys) from e
   

    def run_pipeline(self):
        try:
            self.start_data_ingestion()
            self.start_prepare_base_model()
            self.start_model_trainer()
            self.start_model_evaluation()
        except Exception as e:
            raise exception(e, sys)