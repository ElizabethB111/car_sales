from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xgboost import XGBRegressor
import numpy as np
import os

# Initialize FastAPI
app = FastAPI(title="Car Sales Forecast")

# Load model
MODEL_PATH = "model.json"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Train model first.")
xgb_model = XGBRegressor()
xgb_model.load_model(MODEL_PATH)

# Pydantic schema
class SalesData(BaseModel):
    sales_data: list[float]

@app.post("/predict")
def predict(data: SalesData):
    if len(data.sales_data) != 6:
        raise HTTPException(status_code=400, detail="Provide exactly 6 months of sales data.")

    # Prepare features: last 6 months as lag features, plus rolling mean/std and dummy date features
    lag_features = np.array(data.sales_data[::-1])  # oldest to most recent
    rolling_mean = np.mean(lag_features[-3:])
    rolling_std = np.std(lag_features[-3:])
    
    # For simplicity, use dummy values for Year/Month/Quarter/Day
    features = np.concatenate([
        lag_features,  # sales_lag_1 .. sales_lag_6
        [2025, 11, 4, 1],  # Year, Month_of_Year, Quarter, Day_of_Month
        [rolling_mean, rolling_std]
    ])
    
    prediction = xgb_model.predict(features.reshape(1, -1))[0]
    return {"prediction": round(float(prediction))}
