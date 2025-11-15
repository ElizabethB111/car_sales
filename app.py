from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xgboost import XGBRegressor
import numpy as np
import os

# -----------------------------------------------------------
# Initialize FastAPI
# -----------------------------------------------------------
app = FastAPI(title="Car Sales Forecast API")


# -----------------------------------------------------------
# Load the XGBoost model
# -----------------------------------------------------------
MODEL_PATH = "model.json"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Train the model first.")

xgb_model = XGBRegressor()
xgb_model.load_model(MODEL_PATH)


# -----------------------------------------------------------
# Pydantic schema for POST /predict
# -----------------------------------------------------------
class SalesData(BaseModel):
    sales_data: list[float]


# -----------------------------------------------------------
# Root route — prevents 404s on Render
# -----------------------------------------------------------
@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Car Sales Prediction API is running.",
        "endpoints": {
            "POST /predict": "Send 6 months of sales data to get a forecast."
        }
    }


# -----------------------------------------------------------
# Optional – stops favicon.ico 404 spam in logs
# -----------------------------------------------------------
@app.get("/favicon.ico")
def favicon():
    return {"message": "No favicon"}


# -----------------------------------------------------------
# Prediction endpoint
# -----------------------------------------------------------
@app.post("/predict")
def predict(data: SalesData):
    # Validate input length
    if len(data.sales_data) != 6:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly 6 months of sales data."
        )

    # Convert to numpy array (oldest → newest)
    lag_features = np.array(data.sales_data[::-1])

    # Feature engineering
    rolling_mean = np.mean(lag_features[-3:])
    rolling_std = np.std(lag_features[-3:])

    # Dummy date features (can later be replaced with actual date context)
    year = 2025
    month = 11
    quarter = 4
    day = 1

    # Final feature vector (must match your model training pipeline)
    features = np.concatenate([
        lag_features,         # lag_1 ... lag_6
        [year, month, quarter, day],
        [rolling_mean, rolling_std]
    ])

    # Run model
    prediction = xgb_model.predict(features.reshape(1, -1))[0]

    # Return clean response
    return {
        "prediction": round(float(prediction)),
        "input_used": data.sales_data
    }
