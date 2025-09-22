import os
import sys

from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifacts_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import TrainingPipelineConfig
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.entity.config_entity import ModelTrainerConfig

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Initializing DataIngestionConfig...")
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info(f"DataIngestionConfig initialized: {self.data_ingestion_config}")
            
            logging.info("Initializing DataIngestion...")
            data_ingestion = DataIngestion(self.data_ingestion_config)
            
            logging.info("Starting data ingestion process...")
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            
            if not data_ingestion_artifact:
                raise ValueError("Data ingestion returned None artifact")
                
            logging.info(f"Data ingestion completed successfully: {data_ingestion_artifact}")
            return data_ingestion_artifact
            
        except Exception as e:
            error_msg = f"Error in start_data_ingestion: {str(e)}"
            logging.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
        
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        try:
            logging.info("Initializing DataTransformationConfig...")
            self.data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info(f"DataTransformationConfig initialized: {self.data_transformation_config}")
            
            logging.info("Initializing DataTransformation...")
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config
            )
            
            logging.info("Starting data transformation process...")
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            
            if not data_transformation_artifact:
                raise ValueError("Data transformation returned None artifact")
                
            logging.info(f"Data transformation completed successfully: {data_transformation_artifact}")
            return data_transformation_artifact
            
        except Exception as e:
            error_msg = f"Error in start_data_transformation: {str(e)}"
            logging.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
    
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            self.model_trainer_config=ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config=self.model_trainer_config,data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)
    
    def run_pipeline(self):
        try:
            logging.info("Starting data ingestion...")
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info(f"Data ingestion completed: {data_ingestion_artifact}")
            
            logging.info("Starting data transformation...")
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact)
            logging.info(f"Data transformation completed: {data_transformation_artifact}")
            
            logging.info("Starting model training...")
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            logging.info(f"Model training completed: {model_trainer_artifact}")
            
            return {
                "status": "success",
                "data_ingestion": str(data_ingestion_artifact),
                "data_transformation": str(data_transformation_artifact),
                "model_trainer": str(model_trainer_artifact)
            }
        except Exception as e:
            error_msg = f"Error in pipeline: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return {"status": "error", "message": error_msg}
    
    

    


    

            
