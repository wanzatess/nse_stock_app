from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import os

# ----------------- INIT -----------------
app = FastAPI()

# Load the trained model
model_path = "models/stock_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"{model_path} not found. Train the model first!")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load processed stock data
data_path = "data/processed/NSE_20_stocks_2013_2025_features_target.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found. Make sure your cleaned CSV exists!")

df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)

# ----------------- INPUT -----------------
class StockSymbol(BaseModel):
    symbol: str

# ----------------- PREDICT -----------------
@app.post("/predict")
def predict(symbol_data: StockSymbol):
    symbol = symbol_data.symbol.upper()

    # Filter last day for this stock
    stock_df = df[df['code'] == symbol].sort_values('date', ascending=False).head(1)
    if stock_df.empty:
        raise HTTPException(status_code=404, detail="Stock symbol not found")

    row = stock_df.iloc[0]

    # Prepare features
    features = [[
        row['day_price'],
        row['ma_5'],
        row['ma_10'],
        row['pct_from_12m_low'],
        row['pct_from_12m_high'],
        row['daily_return'],
        row['daily_volatility']
    ]]

    # Predict
    prediction = model.predict(features)[0]

    return {"prediction": prediction}
