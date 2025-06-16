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
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object,get_latest_artifact_dir

from networksecurity.logger.logger import logger
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.pipeline.evaluation_pipeline import ModelEvaluationPipeline
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

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
from fastapi.templating  import Jinja2Templates
templates = Jinja2Templates(directory="templates")

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
    
@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        logger.info(f"Model prediction started for file: {file.filename}")
        
        # Read CSV into DataFrame
        df = pd.read_csv(file.file)

        # Load preprocessing and model
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")

        # Load feature columns
        latest_artifact_dir = get_latest_artifact_dir()
        data_transformation_dir = os.path.join(latest_artifact_dir, "data_transformation")
        feature_columns_file_path = os.path.join(data_transformation_dir, "transformed_object", "feature_columns.pkl")
        feature_columns = load_object(feature_columns_file_path)

        # Ensure input has correct features
        df = df[feature_columns]

        # Create model object
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model, feature_columns=feature_columns)

        # Predict
        y_pred = network_model.predict(df)
        df['predicted_column'] = y_pred

        # Save output CSV
        os.makedirs("prediction_output", exist_ok=True)
        output_path = os.path.join("prediction_output", f"output_{file.filename}")
        df.to_csv(output_path, index=False)

        # Return as HTML
        table_html = df.to_html(classes='table table-striped', index=False)
        logger.info("Model prediction completed successfully.")
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})

    except Exception as e:
        logger.error("Model prediction failed", exc_info=True)
        raise NetworkSecurityException(e, sys)

if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)
