"""
NSE Stock Prediction Service (ULTRA MEMORY OPTIMIZED)
This service loads the model ONLY when prediction is requested, then unloads it
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
import requests
import pickle
import joblib
import gc

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
# PATHS
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "stock_model_compressed.pkl")
HF_MODEL_URL = "https://huggingface.co/wanzatess/nse_stock_model/resolve/main/stock_model_compressed.pkl"

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def download_model_if_needed():
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


def load_and_predict(features: list):
    """
    Load model, make prediction, then immediately unload
    This keeps memory usage low by not keeping model in RAM
    """
    model = None
    try:
        print(f"🔄 Loading model for prediction...")
        
        # Load model
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e1:
            print(f"⚠️ joblib failed: {e1}, trying pickle...")
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
        
        print(f"✅ Model loaded: {type(model)}")
        
        # Make prediction
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)
        raw_pred = prediction[0]
        
        result = None
        
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
        
        print(f"✅ Prediction complete: {pred_value}")
        return result
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None
        
    finally:
        # CRITICAL: Unload model from memory immediately
        if model is not None:
            del model
            gc.collect()
            print(f"🗑️ Model unloaded from memory")


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
        "model_file_exists": os.path.exists(MODEL_PATH),
        "memory_mode": "on-demand (model loads per request)"
    }


@app.get("/health")
def health():
    model_exists = os.path.exists(MODEL_PATH)
    return {
        "status": "ok" if model_exists else "degraded",
        "model_file_exists": model_exists,
        "model_size_mb": os.path.getsize(MODEL_PATH) / 1024 / 1024 if model_exists else 0
    }


@app.get("/model-status")
def model_status():
    model_exists = os.path.exists(MODEL_PATH)
    return {
        "model_file_exists": model_exists,
        "model_path": MODEL_PATH,
        "model_size_mb": os.path.getsize(MODEL_PATH) / 1024 / 1024 if model_exists else 0,
        "mode": "on-demand loading (not kept in memory)"
    }


@app.post("/model-download")
def model_download():
    """Trigger model download"""
    if os.path.exists(MODEL_PATH):
        return {"ok": True, "message": "Model already downloaded"}
    
    success = download_model_if_needed()
    if success:
        return {"ok": True, "message": "Model downloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Model download failed")


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Make prediction by loading model on-demand
    Model is loaded, used, then immediately unloaded to save memory
    """
    if len(request.features) != 7:
        raise HTTPException(
            status_code=400, 
            detail="Expected 7 features: [day_price, ma_5, ma_10, pct_from_12m_low, pct_from_12m_high, daily_return, daily_volatility]"
        )
    
    # Download model if needed
    if not os.path.exists(MODEL_PATH):
        print("📥 Model not found, downloading...")
        if not download_model_if_needed():
            raise HTTPException(status_code=503, detail="Model download failed")
    
    # Load model, predict, unload
    prediction_result = load_and_predict(request.features)
    
    if prediction_result is None:
        raise HTTPException(
            status_code=503, 
            detail="Prediction failed. Model may be corrupted."
        )
    
    response = {
        "prediction": prediction_result,
        "features_used": request.features
    }
    
    if request.symbol:
        response["symbol"] = request.symbol
    
    return response


# ------------------------------
# STARTUP
# ------------------------------
@app.on_event("startup")
async def startup_event():
    print("\n🚀 NSE PREDICTION SERVICE (ULTRA MEMORY OPTIMIZED)")
    print("💾 Memory Strategy: Load model only when needed, unload immediately")
    print("📦 Checking if model file exists...")
    
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
        print(f"✅ Model file found: {size_mb:.2f} MB")
    else:
        print("⚠️ Model file not found - will download on first prediction request")
    
    print("✅ Prediction service ready (model NOT loaded in memory)\n")


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 NSE PREDICTION SERVICE - STARTING UP")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))