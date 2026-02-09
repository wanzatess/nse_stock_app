from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
from datetime import datetime
import requests
import pickle
import gzip
import joblib

# ------------------------------
# FASTAPI APP
# ------------------------------
app = FastAPI()

# ------------------------------
# CORS Setup
# ------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# GLOBAL VARIABLES
# ------------------------------
df = None
last_modified = None
model = None

# ------------------------------
# PATHS
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # goes up one level from backend/
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "NSE_20_stocks_2013_2025_features_target.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "stock_model_compressed.pkl")

# HuggingFace Model URL
HF_MODEL_URL = "https://huggingface.co/wanzatess/nse_stock_model/resolve/main/stock_model_compressed.pkl"

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def load_pickle_file(filepath):
    """
    Intelligently load a pickle file, trying multiple methods
    """
    print(f"📂 Attempting to load: {filepath}")
    
    # Check file signature to determine type
    with open(filepath, 'rb') as f:
        first_bytes = f.read(10)
    
    print(f"🔍 File signature: {first_bytes[:4].hex()}")
    
    # Gzip files start with 0x1f 0x8b
    is_gzipped = (first_bytes[:2] == b'\x1f\x8b')
    
    # Try gzip first
    if is_gzipped:
        print(f"🗜️ Detected gzip-compressed file")
        try:
            with gzip.open(filepath, 'rb') as f:
                loaded_model = pickle.load(f)
            print(f"✅ Loaded with gzip.open + pickle.load")
            return loaded_model
        except Exception as e:
            print(f"⚠️ gzip+pickle failed: {e}")
    
    # Try joblib (handles compression automatically)
    print(f"🔧 Trying joblib.load...")
    try:
        loaded_model = joblib.load(filepath)
        print(f"✅ Loaded with joblib.load")
        return loaded_model
    except Exception as e:
        print(f"⚠️ joblib failed: {e}")
    
    # Try regular pickle
    print(f"📦 Trying regular pickle...")
    try:
        with open(filepath, 'rb') as f:
            loaded_model = pickle.load(f)
        print(f"✅ Loaded with regular pickle")
        return loaded_model
    except Exception as e:
        print(f"⚠️ regular pickle failed: {e}")
    
    raise Exception("All loading methods failed")


def download_model():
    """Download the model from HuggingFace if not present"""
    global model
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Check if model already exists
    if os.path.exists(MODEL_PATH):
        print(f"📦 Found existing model file: {MODEL_PATH}")
        print(f"📊 File size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.2f} MB")
        try:
            model = load_pickle_file(MODEL_PATH)
            print(f"✅ Model loaded successfully!")
            print(f"🔍 Model type: {type(model)}")
            
            # Test the model quickly
            if hasattr(model, 'predict'):
                print(f"✅ Model has predict method")
            if hasattr(model, 'predict_proba'):
                print(f"✅ Model has predict_proba method")
            
            return
        except Exception as e:
            print(f"⚠️ Failed to load existing model: {e}")
            print(f"🔍 Error type: {type(e).__name__}")
            print("🗑️ Deleting corrupted file and re-downloading...")
            try:
                os.remove(MODEL_PATH)
            except:
                pass
    
    # Download model from HuggingFace with proper headers
    print(f"⬇️ Downloading model from HuggingFace...")
    print(f"🔗 URL: {HF_MODEL_URL}")
    
    try:
        # Add headers to ensure we get the raw file, not HTML
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/octet-stream'
        }
        
        response = requests.get(HF_MODEL_URL, headers=headers, timeout=120, stream=True)
        response.raise_for_status()
        
        # Check if we got HTML instead of binary data
        content_type = response.headers.get('Content-Type', '')
        print(f"📋 Content-Type: {content_type}")
        
        if 'text/html' in content_type:
            print(f"❌ Received HTML instead of binary file")
            print(f"💡 Trying with ?download=true parameter...")
            response = requests.get(f"{HF_MODEL_URL}?download=true", headers=headers, timeout=120, stream=True)
            response.raise_for_status()
        
        # Save the model
        print(f"💾 Saving model to: {MODEL_PATH}")
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size = os.path.getsize(MODEL_PATH)
        print(f"✅ Model downloaded successfully!")
        print(f"📊 File size: {file_size / 1024 / 1024:.2f} MB")
        
        # Verify the download
        with open(MODEL_PATH, 'rb') as f:
            first_bytes = f.read(10)
            print(f"🔍 File signature: {first_bytes[:4].hex()}")
        
        # Load the model using intelligent loader
        model = load_pickle_file(MODEL_PATH)
        print(f"✅ Model loaded successfully!")
        print(f"🔍 Model type: {type(model)}")
        
        # Test the model
        if hasattr(model, 'predict'):
            print(f"✅ Model has predict method")
        if hasattr(model, 'predict_proba'):
            print(f"✅ Model has predict_proba method")
        
    except Exception as e:
        print(f"❌ Failed to download/load model: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        print(f"⚠️ Predictions will not be available")
        print(f"")
        print(f"💡 SOLUTION:")
        print(f"   1. Manually compress your 200MB model:")
        print(f"      python compress_model.py")
        print(f"   2. Upload to HuggingFace: {HF_MODEL_URL}")
        print(f"   3. Restart this service")
        
        # Debug: show first bytes if file exists
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, 'rb') as f:
                    first_bytes = f.read(100)
                    print(f"🔍 First bytes: {first_bytes[:50]}")
            except:
                pass
        
        model = None


def reload_data_if_needed():
    global df, last_modified
    if not os.path.exists(DATA_PATH):
        print(f"⚠️ Data file not found: {DATA_PATH}")
        return

    current_modified = os.path.getmtime(DATA_PATH)
    if last_modified is None or current_modified > last_modified:
        print(f"🔄 Loading data from: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH, low_memory=False)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        last_modified = current_modified
        print(f"✅ Data loaded! {len(df)} records, {df['code'].nunique()} stocks")


def get_latest_stock_data(symbol: str):
    reload_data_if_needed()
    if df is None:
        return None

    stock_df = df[df["code"] == symbol].sort_values("date", ascending=False)
    if stock_df.empty:
        return None

    row = stock_df.iloc[0]
    change_pct = row["changepct"]
    if isinstance(change_pct, str):
        change_pct = float(change_pct.replace("%", ""))

    return {
        "symbol": symbol,
        "name": row["name"],
        "current_price": float(row["day_price"]),
        "previous_price": float(row["previous"]) if pd.notna(row["previous"]) else float(row["day_price"]),
        "change": float(row["change"]) if pd.notna(row["change"]) else 0.0,
        "change_percent": float(change_pct) if pd.notna(change_pct) else 0.0,
        "day_low": float(row["day_low"]) if pd.notna(row["day_low"]) else float(row["day_price"]),
        "day_high": float(row["day_high"]) if pd.notna(row["day_high"]) else float(row["day_price"]),
        "volume": int(row["volume"]) if pd.notna(row["volume"]) else 0,
        "ma_5": float(row["ma_5"]) if pd.notna(row["ma_5"]) else float(row["day_price"]),
        "ma_10": float(row["ma_10"]) if pd.notna(row["ma_10"]) else float(row["day_price"]),
        "pct_from_12m_low": float(row["pct_from_12m_low"]) if pd.notna(row["pct_from_12m_low"]) else 0.0,
        "pct_from_12m_high": float(row["pct_from_12m_high"]) if pd.notna(row["pct_from_12m_high"]) else 0.0,
        "daily_return": float(row["daily_return"]) if pd.notna(row["daily_return"]) else 0.0,
        "daily_volatility": float(row["daily_volatility"]) if pd.notna(row["daily_volatility"]) else 0.0,
        "last_updated": row["date"].strftime("%Y-%m-%d"),
    }


def predict_with_model(features: list):
    """Make prediction using the loaded model"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please try again later."
        )
    
    try:
        # Convert features to numpy array with shape (1, n_features)
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_array)
            
            # Assuming binary or multi-class classification
            # Adjust based on your model's output
            return {
                "prediction": int(prediction[0]),
                "confidence": float(max(probabilities[0])),
                "probabilities": probabilities[0].tolist()
            }
        else:
            # For regression or models without predict_proba
            return {
                "prediction": float(prediction[0]),
                "confidence": None,
                "probabilities": None
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ------------------------------
# REQUEST MODELS
# ------------------------------
class PredictRequest(BaseModel):
    symbol: str

# ------------------------------
# API ROUTES
# ------------------------------
@app.get("/")
def root():
    reload_data_if_needed()
    return {
        "status": "healthy" if df is not None else "degraded",
        "data_loaded": df is not None,
        "model_loaded": model is not None,
        "total_stocks": df["code"].nunique() if df is not None else 0,
    }


@app.get("/stocks")
def get_stocks():
    reload_data_if_needed()
    if df is None:
        raise HTTPException(status_code=503, detail="Data