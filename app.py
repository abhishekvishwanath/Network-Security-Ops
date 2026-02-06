import os
import sys

import certifi
import pymongo
import pandas as pd
from dotenv import load_dotenv

from networksecurity.utils.network_model import NetworkSecurityModel
from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging
from networksecurity.pipelines.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils import load_object
from networksecurity.constants import DATA_INGESTION_DATABASE_NAME, DATA_INGESTION_COLLECTION_NAME

# Type checking: These imports are available at runtime
# Install packages: pip install fastapi uvicorn starlette
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi import FastAPI, File, UploadFile, Request  # type: ignore
from uvicorn import run as app_run  # type: ignore
from starlette.responses import RedirectResponse  # type: ignore
from fastapi.templating import Jinja2Templates  # type: ignore

# Load environment variables
load_dotenv()

# MongoDB connection setup (for potential future use)
mongo_db_url = os.getenv('MONGO_DB_URL')
if mongo_db_url:
    certifi.where()  # Ensure certifi is available for pymongo SSL
    client = pymongo.MongoClient(mongo_db_url)
    database = client[DATA_INGESTION_DATABASE_NAME]
    collection = database[DATA_INGESTION_COLLECTION_NAME]

# FastAPI app setup
templates = Jinja2Templates(directory='./templates')

app = FastAPI()
origin = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origin,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/', tags=['authentication'])
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

@app.post('/predict')
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        preprocessor = load_object('final_model/preprocessor.pkl')
        final_model = load_object('final_model/model.pkl')
        network_security_model = NetworkSecurityModel(preprocessor, final_model)
        
        # Predict for all rows in the dataframe
        y_pred = network_security_model.predict(df)
        
        # Handle single prediction case
        if len(y_pred) == 1:
            df["predicted_column"] = y_pred[0]
        else:
            df["predicted_column"] = y_pred
        print(df["predicted_column"])
        df.to_csv('prediction_output/output.csv', index=False)

        table_html = df.to_html(classes="table table-striped")
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
        
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    # Allow port to be specified via command line argument
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    app_run(app, host='0.0.0.0', port=port)