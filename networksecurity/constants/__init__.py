import os
import sys
import numpy as np  
import pandas as pd


PIPELINE_NAME = "Network Security"
ARTIFACT_DIR = "artifact"
FILE_NAME = "phisingData.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TARGET_COLUMN = "Result"

DATA_INGESTION_COLLECTION_NAME = "NetworkTrafficData"
DATA_INGESTION_DATABASE_NAME = "NetworkSecurity"
DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION = 0.2

DATA_TRANSFORMATION_DIR = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR = "transformed"
DATA_TRANSFORMATION_TRANSFORMER_OBJECT = "preprocessor.pkl"
DATA_TRANSFORMER_IMPUTER_PARAMS = {"missing_values":np.nan,
                                   "n_neighbors":3,
                                   "weights":"uniform"}

MODEL_TRAINER_DIR = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE = 0.6
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD = 0.05

