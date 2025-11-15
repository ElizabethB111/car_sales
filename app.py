from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from xgboost import XGBRegressor
import numpy as np
import os

# -----------------------------------------------------------
# Initialize FastAPI
# -----------------------------------------------------------
app = FastAPI(title="Car Sales Forecast API")

# -----------------------------------------------------------
# Serve frontend folder (index.html lives here)
# -----------------------------------------------------------
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


# -----------------------------------------------------------
# Load XGBoost model
# -----------------------------------------------------------
MODEL_PATH = "model.json"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("model.json not found — run train_model.py")

xgb_model = XGBRegressor()
xgb_model.load_model(MODEL_PATH)


# -----------------------------------------------------------
# Pydantic Model for API Input
# -----------------------------------------------------------
class SalesData(BaseModel):
    sales_data: list[float]


# -----------------------------------------------------------
# Home Route → serves index.html form
# -----------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    try:
        with open("frontend/index.html", "r") as f:
            return f.read()
    except Exception as e:
        return f"<h1>Error loading index.html</h1><p>{e}</p>"


# -----------------------------------------------------------
# Optional — avoid favicon errors
# -----------------------------------------------------------
@app.get("/favicon.ico")
def favicon():
    return {"message": "no favicon"}


# -----------------------------------------------------------
# Prediction Endpoint
# -----------------------------------------------------------
@app.post("/predict")
def predict(data: SalesData):

    # Must be exactly six months
    if len(data.sales_data) != 6:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly 6 months of sales data."
        )

    # Reverse to match training lag order  
    # (model expects lag_1 = last month, lag_6 = oldest)
    lag_features = np.array(data.sales_data[::-1])

    # Rolling features (same as training)
    rolling_mean = np.mean(lag_features[-3:])
    rolling_std = np.std(lag_features[-3:])

    # Static date features (training used actual dates)
    year = 2025
    month = 11
    quarter = 4
    day = 1

    # Final feature vector (must match train_model.py)
    features = np.concatenate([
        lag_features,
        [year, month, quarter, day],
        [rolling_mean, rolling_std]
    ])

    # Generate prediction
    prediction = xgb_model.predict(features.reshape(1, -1))[0]

    return {
        "prediction": round(float(prediction)),
        "input_used": data.sales_data
    }

