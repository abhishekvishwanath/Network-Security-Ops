import os 
import sys

import pandas as pd
import pymongo
from pymongo import MongoClient
import json
from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging
from dotenv import load_dotenv

load_dotenv()

mongo_db_url = os.getenv("MONGODB_URI")

import certifi
ca = certifi.where()

class NetworkDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)
    
    def csv_to_json_converter(slf,file_path):
        """
        This function converts a CSV file to a JSON format and returns it as a pandas DataFrame.
        
        Args:
        file_path (str): The path to the CSV file.
        
        Returns:
        pd.DataFrame: A DataFrame containing the JSON data.
        """
        try:
            df = pd.read_csv(file_path)
            df.reset_index(drop=True,inplace=True)
            records=list(json.loads(df.T.to_json()).values())
            return records
        except Exception as e:
            raise CustomException(e,sys)
    
    def insert_data_to_mongodb(self,records:pd.DataFrame,db_name:str,collection_name:str):
        """
        Inserts data into a MongoDB collection.
        
        Args:
        records (pd.DataFrame): The data to be inserted.
        db_name (str): The name of the database.
        collection_name (str): The name of the collection.
        
        Returns:
        None
        """
        try:
            self.records=records
            self.db_name=db_name
            self.collection_name=collection_name
            self.client = pymongo.MongoClient(mongo_db_url)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            self.collection.insert_many(self.records)
            logging.info("Data inserted successfully into MongoDB")
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    try:
        ne = NetworkDataExtract()
        records = ne.csv_to_json_converter(file_path="data/phisingData.csv")
        ne.insert_data_to_mongodb(records=records,db_name="NetworkSecurity",collection_name="NetworkTrafficData")
    except Exception as e:
        raise CustomException(e,sys)

