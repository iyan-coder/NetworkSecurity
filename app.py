import sys
import os
import certifi
from dotenv import load_dotenv
import pymongo
import pandas as pd
import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from networksecurity.logger.logger import logger
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.pipeline.evaluation_pipeline import ModelEvaluationPipeline

from networksecurity.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME, DATA_INGESTION_COLLECTION_NAME

# === Load environment variables ===
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL_KEY")
ca = certifi.where()

# === MongoDB setup ===
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# === FastAPI App ===
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# === Routes ===

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train", tags=["pipeline"])
async def train_route():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_training_pipeline()
        return {"message": "Training completed successfully!"}
    except Exception as e:
        logger.error("Training failed", exc_info=True)
        raise NetworkSecurityException(e, sys)


@app.get("/evaluate", tags=["pipeline"])
async def evaluate_route():
    try:
        evaluation_pipeline = ModelEvaluationPipeline()
        evaluation_pipeline.run_evaluation_pipeline()
        return {"message": "Evaluation completed successfully!"}
    except Exception as e:
        logger.error("Evaluation failed", exc_info=True)
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
