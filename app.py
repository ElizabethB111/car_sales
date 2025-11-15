# app.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, conlist
import os
import xgboost as xgb
import numpy as np
from datetime import datetime, timedelta
import calendar

# --- Request model ---
class PredictRequest(BaseModel):
    # sales_data: most recent first, length exactly 6
    sales_data: conlist(float, min_items=6, max_items=6)

# --- App setup ---
app = FastAPI(title="Car Sales Forecast")

# Serve static frontend (frontend/index.html)
if not os.path.exists("frontend"):
    os.makedirs("frontend", exist_ok=True)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

# We'll respond to GET / with the index.html
@app.get("/", response_class=FileResponse)
def read_index():
    index_path = os.path.join("frontend", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return index_path

# --- Load model once at startup ---
MODEL_PATH = os.path.join("model.json")
booster = None

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    b = xgb.Booster()
    # xgboost.Booster().load_model accepts JSON model file saved with save_model(...)
    b.load_model(path)
    return b

@app.on_event("startup")
def startup_event():
    global booster
    try:
        booster = load_model()
        print(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        print("Failed to load model on startup:", e)
        # allow app to start but /predict will return error if model missing

# --- Helper: build feature vector ---
def build_features_from_sales(sales_list):
    """
    sales_list: [most_recent, month2, month3, month4, month5, month6]
    We'll construct features in the same order used during training:
       sales_lag_1 .. sales_lag_6 (lag_1 = most recent)
       Year, Month_of_Year, Quarter, Day_of_Month
    For time features we set them based on the month we are predicting (i.e., next month).
    """
    # sales_lag_1..6: convert to floats
    sales_lags = [float(x) for x in sales_list]

    # Determine the "prediction month" as the month after the most recent one.
    # We do not get dates from the frontend, so use today's date as reference: assume
    # the most_recent input corresponds to the last completed month. For example,
    # if today is Nov 15 2025, assume user entered Oct 2025 as most recent, so predict Nov 2025.
    # Implementation: use current date and add one month.
    today = datetime.utcnow().date()
    # Prediction month = today + 1 month
    year = today.year
    month = today.month + 1
    if month == 13:
        year += 1
        month = 1

    # day-of-month for monthly data - use 1 (consistent with training where Month likely had day=1)
    day_of_month = 1
    quarter = (month - 1) // 3 + 1

    # Compose features in the same order as training
    feature_vector = sales_lags + [year, month, quarter, day_of_month]
    return np.array(feature_vector, dtype=float).reshape(1, -1)

# --- Prediction endpoint ---
@app.post("/predict")
async def predict(req: PredictRequest):
    global booster
    if booster is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server")

    try:
        feats = build_features_from_sales(req.sales_data)
        # XGBoost Booster requires DMatrix for predict
        dmat = xgb.DMatrix(feats)
        pred = booster.predict(dmat)
        # booster.predict returns array-like
        prediction = float(pred[0])
        return JSONResponse({"prediction": prediction})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Optional health check ---
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": booster is not None}
