from pathlib import Path
import numpy as np
from fastapi import FastAPI, Response, HTTPException, status
from joblib import load
from .schemas import Wine, Rating, feature_names
from azureml.core import Workspace
from azureml.core.model import Model

ROOT_DIR = Path(__file__).parent.parent

app = FastAPI()
# scaler = load(ROOT_DIR / "artifacts/scaler.joblib")
# model = load(ROOT_DIR / "artifacts/model.joblib")

ws = Workspace(subscription_id="0dfd6360-d4a6-4d90-b642-22bc52ee4a2b",
               resource_group="azure-mlops",
               workspace_name="ml-pipeline")

@app.get("/")
def root():
    return "Wine Quality Ratings !!"

@app.post("/predict", response_model=Rating)
def predict(response: Response, sample: Wine, model_name: str, version: int):
    sample_dict = sample.dict()
    features = np.array([sample_dict[f] for f in feature_names]).reshape(1, -1)
    model = Model(
        ws, 
        f"{model_name}_model", 
        version=version)
    scaler = Model(
        ws, 
        f"{model_name}_scaler", 
        version=version
    )
    try:
        model.download(target_dir="artifacts/", exist_ok= True)
        scaler.download(target_dir="artifacts/", exist_ok= True)
        scaler = load(ROOT_DIR / f'artifacts/{model_name}_scaler.joblib')
        model = load(ROOT_DIR / f'artifacts/{model_name}_model.joblib')
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        response.headers["X-model-score"] = str(prediction)
        return Rating(quality=prediction)
    except:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This model is not exists",
        )   

@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}