from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from xgboost import XGBRegressor
import numpy as np
import os


# -----------------------------------------------------------
# Initialize FastAPI
# -----------------------------------------------------------
app = FastAPI(title="Car Sales Forecast API")


# -----------------------------------------------------------
# Load XGBoost model
# -----------------------------------------------------------
MODEL_PATH = "model.json"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Train the model first.")

xgb_model = XGBRegressor()
xgb_model.load_model(MODEL_PATH)


# -----------------------------------------------------------
# Data Model
# -----------------------------------------------------------
class SalesData(BaseModel):
    sales_data: list[float]


# -----------------------------------------------------------
# HTML Home Page
# -----------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Car Sales Forecast API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 40px;
                    text-align: center;
                }
                .card {
                    max-width: 700px;
                    margin: auto;
                    background: white;
                    padding: 40px;
                    border-radius: 12px;
                    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
                }
                h1 {
                    color: #333;
                    margin-bottom: 10px;
                }
                p {
                    font-size: 18px;
                    color: #555;
                }
                code {
                    background: #eee;
                    display: block;
                    padding: 12px;
                    border-radius: 8px;
                    margin-top: 20px;
                    text-align: left;
                    font-size: 15px;
                }
                .footer {
                    margin-top: 30px;
                    font-size: 14px;
                    color: #666;
                }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>🚗 Car Sales Forecast API</h1>
                <p>This API uses machine learning (XGBoost) to forecast future vehicle sales.</p>

                <p><strong>Prediction Endpoint:</strong></p>
                <code>POST /predict</code>

                <p>Send exactly 6 months of sales data:</p>
                <code>
                {
                    "sales_data": [120, 135, 150, 160, 170, 185]
                }
                </code>

                <p class="footer">Created with FastAPI · Hosted on Render</p>
            </div>
        </body>
    </html>
    """


# -----------------------------------------------------------
# Optional – stop favicon errors in Render logs
# -----------------------------------------------------------
@app.get("/favicon.ico")
def favicon():
    return {"message": "No favicon available"}


# -----------------------------------------------------------
# Prediction Endpoint
# -----------------------------------------------------------
@app.post("/predict")
def predict(data: SalesData):
    # Validate input length
    if len(data.sales_data) != 6:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly 6 months of sales data."
        )

    # Convert to numpy array (old → new)
    lag_features = np.array(data.sales_data[::-1])

    # Rolling stats
    rolling_mean = np.mean(lag_features[-3:])
    rolling_std = np.std(lag_features[-3:])

    # Static date features (can be replaced later)
    year = 2025
    month = 11
    quarter = 4
    day = 1

    # Combine all features (must match training)
    features = np.concatenate([
        lag_features,         # Lag 6 months
        [year, month, quarter, day],
        [rolling_mean, rolling_std]
    ])

    prediction = xgb_model.predict(features.reshape(1, -1))[0]

    return {
        "prediction": round(float(prediction)),
        "input_used": data.sales_data
    }
