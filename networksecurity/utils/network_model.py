import os
import sys

from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging

from networksecurity.constants import MODEL_TRAINER_TRAINED_MODEL_NAME,MODEL_TRAINER_DIR


class NetworkSecurityModel:
    def __init__(self,preprocessor,classifier):
        try:
            self.preprocessor=preprocessor
            self.classifier=classifier
        except Exception as e:
            raise CustomException(e,sys)
    
    def predict(self,data):
        try:
            logging.info("Starting prediction process")
            transformed_data=self.preprocessor.transform(data)
            logging.info("Data preprocessing completed")
            y=self.classifier.predict(transformed_data)
            logging.info(f"Prediction completed for {len(data)} samples")
            return y
        except Exception as e:
            raise CustomException(e,sys)
    
    
    

    

