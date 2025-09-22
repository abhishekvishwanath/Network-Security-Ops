import os
import sys

import certifi

from networksecurity.utils import network_model
from push_mongodb import mongo_db_url
ca=certifi.where()

from dotenv import load_dotenv
load_dotenv()

mongo_db_url=os.getenv('MONGO_DB_URL')
print(mongo_db_url)

import pymongo 
import pandas as pd

from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging
from networksecurity.pipelines.training_pipeline import TrainingPipeline

from networksecurity.utils.main_utils import load_object

from networksecurity.constants import DATA_INGESTION_DATABASE_NAME,DATA_INGESTION_COLLECTION_NAME

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI,File,UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse

from fastapi.templating import Jinja2Templates
templates=Jinja2Templates(directory='./templates')

client= pymongo.MongoClient(mongo_db_url)
database=client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app=FastAPI()
origin=["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origin,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/',tags=['authentication'])
async def index():
    return RedirectResponse(url='/docs')

@app.get('/train')
async def train():
    try:
        logging.info("Starting training pipeline...")
        training_pipeline = TrainingPipeline()
        result = training_pipeline.run_pipeline()
        return {"status": "success", "message": "Training completed successfully", "details": str(result)}
    except Exception as e:
        error_message = f"Error during training: {str(e)}"
        logging.error(error_message)
        return {"status": "error", "message": "Training failed", "error": error_message}

@app.get('/predict')
async def predict(request: Request,file: UploadFile = File(...)):
    try:
        df=pd.read_csv(file.file)
        preprocessor=load_object('preprocessor.pkl')
        final_model=load_object('final_model.pkl')
        network_model=network_model(preprocessor,final_model)
        
        y_pred=network_model.predict(df.iloc[0])
        print(y_pred)

        df["predicted_column"]=y_pred
        print(df["predicted_column"])
        df.to_csv('prediction_output/output.csv',index=False)

        table_html=df.to_html(classes="table table-striped")
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
        
    except Exception as e:
        raise CustomException(e,sys)

if __name__ == "__main__":
    app_run(app,host='0.0.0.0', port=8000)