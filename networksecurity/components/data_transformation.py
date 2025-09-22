import os
import sys
import pandas as pd
import numpy as np
from networksecurity.exception.exception import CustomException
from networksecurity.entity.artifacts_entity import DataIngestionArtifact
from networksecurity.entity.artifacts_entity import DataTransformationArtifact
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.constants import TARGET_COLUMN
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils import save_numpy_array_data, save_object
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


class DataTransformation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_transformation_config:DataTransformationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomException(e,sys)
    
    def read_csv(self,file_path:str)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e,sys)
    
    def get_data_transformation_object(self, numeric_columns):
        try:
            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', KNNImputer(n_neighbors=3)),
                ('scaler', StandardScaler())
            ])
            
            # Create preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numeric_columns)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            train_df = self.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = self.read_csv(self.data_ingestion_artifact.test_file_path)

            # Get numeric columns (excluding target)
            numeric_columns = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if TARGET_COLUMN in numeric_columns:
                numeric_columns.remove(TARGET_COLUMN)

            input_features_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            input_features_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Get and fit preprocessor
            preprocessor = self.get_data_transformation_object(numeric_columns)
            
            # Transform the data
            transformed_train_features = preprocessor.fit_transform(input_features_train_df)
            transformed_test_features = preprocessor.transform(input_features_test_df)
            
            # Save the preprocessor
            save_object("final_model/preprocessor.pkl",preprocessor)
            
            # Combine features and target
            train_arr = np.c_[transformed_train_features, target_feature_train_df.values]
            test_arr = np.c_[transformed_test_features, target_feature_test_df.values]
            
            # Save the transformed data
            save_numpy_array_data(train_arr, self.data_transformation_config.transformed_train_file_path)
            save_numpy_array_data(test_arr, self.data_transformation_config.transformed_test_file_path)
            save_object(self.data_transformation_config.transformed_object_file_path,preprocessor)

            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact
            
        except Exception as e:
            raise CustomException(e,sys)
            

    
