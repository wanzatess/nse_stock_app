"""
NSE Stock Prediction Service (MODEL LOADED AT STARTUP)
Loads model once at startup and keeps it in memory
Should work now that data service only uses 150MB
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
import requests
import pickle
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
model = None  # Will be loaded at startup

# ------------------------------
# PATHS
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "stock_model_compressed.pkl")
HF_MODEL_URL = "https://huggingface.co/wanzatess/nse_stock_model/resolve/main/stock_model_compressed.pkl"

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def download_model():
    """Download model from HuggingFace if not present"""
    if os.path.exists(MODEL_PATH):
        print(f"📦 Model file exists: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.2f} MB")
        return True
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    print(f"⬇️ Downloading model from HuggingFace...")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/octet-stream'
        }
        
        response = requests.get(HF_MODEL_URL, headers=headers, timeout=120, stream=True)
        response.raise_for_status()
        
        print(f"💾 Saving model to: {MODEL_PATH}")
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size = os.path.getsize(MODEL_PATH)
        print(f"✅ Model downloaded: {file_size / 1024 / 1024:.2f} MB")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False


def load_model():
    """Load model into memory"""
    global model
    
    print(f"📂 Loading model from: {MODEL_PATH}")
    
    try:
        # Try joblib first
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded with joblib: {type(model)}")
    except Exception as e1:
        print(f"⚠️ joblib failed: {e1}")
        try:
            # Try pickle
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print(f"✅ Model loaded with pickle: {type(model)}")
        except Exception as e2:
            print(f"❌ All loading methods failed")
            print(f"   joblib error: {e1}")
            print(f"   pickle error: {e2}")
            model = None
            return False
    
    # Test the model
    if model is not None:
        if hasattr(model, 'predict'):
            print(f"✅ Model has predict method")
        if hasattr(model, 'predict_proba'):
            print(f"✅ Model has predict_proba method")
        return True
    
    return False


# ------------------------------
# REQUEST MODELS
# ------------------------------
class PredictRequest(BaseModel):
    features: list
    symbol: str = None


# ------------------------------
# API ROUTES
# ------------------------------
@app.get("/")
def read_root():
    return {
        "service": "NSE Stock Prediction Service",
        "status": "active",
        "model_loaded": model is not None,
        "memory_mode": "persistent (model loaded at startup)"
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_type": str(type(model)) if model else None
    }


@app.get("/model-status")
def model_status():
    return {
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "model_file_exists": os.path.exists(MODEL_PATH),
        "mode": "persistent (loaded at startup)"
    }


@app.get("/predict")
def predict_get_info():
    return {
        "message": "Use POST method",
        "usage": {
            "method": "POST",
            "body": {"features": [100, 102, 103, 15, -5, 0.02, 0.015], "symbol": "ABSA"}
        }
    }


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Make prediction using pre-loaded model
    Fast because model is already in memory
    """
    if len(request.features) != 7:
        raise HTTPException(
            status_code=400, 
            detail="Expected 7 features: [day_price, ma_5, ma_10, pct_from_12m_low, pct_from_12m_high, daily_return, daily_volatility]"
        )
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Check /model-status for details."
        )
    
    try:
        # Make prediction (fast since model is already loaded)
        features_array = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features_array)
        raw_pred = prediction[0]
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_array)
            conf = float(max(probabilities[0]))
            
            try:
                pred_value = int(raw_pred)
            except Exception:
                try:
                    pred_value = float(raw_pred)
                except Exception:
                    pred_value = str(raw_pred)
            
            result = {
                "prediction": pred_value,
                "confidence": conf,
                "probabilities": probabilities[0].tolist()
            }
        else:
            try:
                pred_value = float(raw_pred)
            except Exception:
                pred_value = str(raw_pred)
            
            result = {
                "prediction": pred_value,
                "confidence": None,
                "probabilities": None
            }
        
        response = {
            "prediction": result,
            "features_used": request.features
        }
        
        if request.symbol:
            response["symbol"] = request.symbol
        
        return response
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )


# ------------------------------
# STARTUP
# ------------------------------
@app.on_event("startup")
async def startup_event():
    global model
    
    print("\n🚀 NSE PREDICTION SERVICE - STARTING UP")
    print("💾 Memory Strategy: Load model at startup, keep in memory")
    print("📊 This should work now that data service only uses 150MB\n")
    
    # Download model if needed
    if not os.path.exists(MODEL_PATH):
        print("📥 Model file not found, downloading...")
        if not download_model():
            print("❌ STARTUP FAILED: Could not download model")
            print("⚠️ Service will start but predictions will fail")
            return
    
    # Load model into memory
    print("🔄 Loading model into memory...")
    if load_model():
        print("✅ Model loaded successfully!")
        print(f"📊 Model ready for predictions")
    else:
        print("❌ STARTUP FAILED: Could not load model")
        print("⚠️ Service will start but predictions will fail")
    
    print("\n✅ Prediction service ready\n")


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 NSE PREDICTION SERVICE - STARTING UP")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))