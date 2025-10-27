from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="AI Smart Indian Traffic Predictor")

# Load model at startup (ensure traffic_model.pkl is in same folder)
model = joblib.load("traffic_model.pkl")

class Input(BaseModel):
    day_of_week: int
    hour: int
    temperature: float
    weather: int
    vehicle_count: float

@app.get("/")
def home():
    return {"message": "AI Smart Indian Traffic Model is live"}

@app.post("/predict")
def predict(data: Input):
    X = np.array([[data.day_of_week, data.hour, data.temperature, data.weather, data.vehicle_count]])
    pred = model.predict(X)[0]
    return {"prediction": pred}
