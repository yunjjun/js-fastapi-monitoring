from pathlib import Path
import numpy as np
from fastapi import FastAPI, Response
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
model = Model(
    ws, 
    'wine_model', 
    version=3)
scaler = Model(
    ws, 
    'wine_scaler', 
    version=3
)

@app.get("/")
def root():
    return "Wine Quality Ratings !!"


@app.post("/predict", response_model=Rating)
def predict(response: Response, sample: Wine):
    sample_dict = sample.dict()
    features = np.array([sample_dict[f] for f in feature_names]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    response.headers["X-model-score"] = str(prediction)
    return Rating(quality=prediction)


@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}