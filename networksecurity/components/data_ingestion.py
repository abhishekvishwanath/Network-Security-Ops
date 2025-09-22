import os
import sys
from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifacts_entity import DataIngestionArtifact

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pymongo
from typing import List
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGODB_URI")

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise CustomException(e,sys)
        
    def export_collection_as_dataframe(self)->pd.DataFrame:
        try:
            logging.info("Exporting data from MongoDB to DataFrame")
            database_name=self.data_ingestion_config.database_name
            collection_name=self.data_ingestion_config.collection_name
            mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            collection=mongo_client[database_name][collection_name]
            df= pd.DataFrame(list(collection.find()))
            if "_id" in df.columns:
                df=df.drop(columns=["_id"],axis=1)
            df.replace(to_replace="na",value=np.nan,inplace=True)
            return df
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def export_data_to_feature_store(self,df:pd.DataFrame)->None:
        try:
            logging.info("Exporting data to feature store")
            feature_store_dir=os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir,exist_ok=True)
            df.to_csv(self.data_ingestion_config.feature_store_file_path,index=False,header=True)
            return df
        except Exception as e:
            raise CustomException(e,sys)
    
    def split_data_as_train_test(self,df:pd.DataFrame)->None:
        try:
            logging.info("Splitting data into train and test sets")
            train_set,test_set=train_test_split(df,test_size=self.data_ingestion_config.test_size,random_state=42)
            train_dir=os.path.dirname(self.data_ingestion_config.train_file_path)
            test_dir=os.path.dirname(self.data_ingestion_config.test_file_path)
            os.makedirs(train_dir,exist_ok=True)
            os.makedirs(test_dir,exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.train_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_file_path,index=False,header=True)
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_ingestion(self)->None:
        try:
            df=self.export_collection_as_dataframe()
            df=self.export_data_to_feature_store(df)
            self.split_data_as_train_test(df)
            data_ingestion_artifact=DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )
            logging.info(f"Data Ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e,sys)
    

    

    
