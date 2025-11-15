import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Helper functions
def create_lag_features(df, lags=6):
    df = df.copy()
    for i in range(1, lags + 1):
        df[f'sales_lag_{i}'] = df['Sales'].shift(i)
    return df

def create_rolling_features(df, window_size=3):
    df = df.copy()
    df[f'sales_rolling_mean_{window_size}'] = df['Sales'].shift(1).rolling(window_size).mean()
    df[f'sales_rolling_std_{window_size}'] = df['Sales'].shift(1).rolling(window_size).std()
    return df

def train_car_sales_model(df):
    df = df.copy()

    # Convert month to datetime
    df['Month'] = pd.to_datetime(df['Month'])
    df = df.sort_values('Month')

    # Time-based features
    df['Year'] = df['Month'].dt.year
    df['Month_of_Year'] = df['Month'].dt.month
    df['Quarter'] = df['Month'].dt.quarter
    df['Day_of_Month'] = df['Month'].dt.day  # always 1 for monthly data

    # Lag features
    df = create_lag_features(df, lags=6)

    # Rolling features
    df = create_rolling_features(df, window_size=3)

    # Drop NaN rows from lag/rolling
    df = df.dropna()

    # Features and target
    feature_cols = [f'sales_lag_{i}' for i in range(1,7)] + \
                   ['Year','Month_of_Year','Quarter','Day_of_Month'] + \
                   ['sales_rolling_mean_3','sales_rolling_std_3']

    X = df[feature_cols]
    y = df['Sales']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Train XGBoost
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model trained successfully!")
    print(f"Mean Absolute Error: {mae:.2f}")

    # Save model to root-level model.json
    model.get_booster().save_model("model.json")
    print("Model saved as model.json")

    return model, X.columns.tolist()


if __name__ == "__main__":
    df = pd.read_csv("monthly-car-sales.csv")
    model, features = train_car_sales_model(df)
