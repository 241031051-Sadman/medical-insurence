from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app = FastAPI(title="Mall Customer Segmentation Engine")

class CustomerMetrics(BaseModel):
    Age: int
    AnnualIncome: int
    SpendingScore: int

scaler = joblib.load('models/scaler.pkl')
kmeans_model = joblib.load('models/complex_model.pkl')

persona_map = {
    0: "Sensible Spenders (Middle Income, Moderate Spending)",
    1: "Target Cohort (High Income, High Spending)",
    2: "Careful Earners (High Income, Low Spending)",
    3: "Reckless Buyers (Low Income, High Spending)",
    4: "Conservative Profiles (Low Income, Low Spending)"
}

@app.post("/predict")
def generate_segmentation(metrics: CustomerMetrics):
    input_df = pd.DataFrame([metrics.model_dump()])
    
    scaled_data = scaler.transform(input_df)
    cluster_id = int(kmeans_model.predict(scaled_data)[0])
    
    return {
        "status": "Success",
        "cluster_assignment": cluster_id,
        "customer_persona": persona_map.get(cluster_id, "Unknown Segment Profile")
    }
@app.get("/")
def home_health_check():
    return {
        "status": "Online",
        "project": "NLP Toxicity Classification Engine",
        "version": "1.0.0"
    }
