import joblib
import os

# Path to your original model
original_model_path = "stock_model.pkl"

# Path for compressed model
compressed_model_path = "stock_model_compressed.pkl"

# Check if original model exists
if not os.path.exists(original_model_path):
    raise FileNotFoundError(f"{original_model_path} not found!")

# Load the original model
print("Loading original model...")
model = joblib.load(original_model_path)
print("Model loaded successfully!")

# Save with compression
compression_level = 3  # Try 3–9; higher = smaller file but slower load
print(f"Compressing model (level {compression_level})...")
joblib.dump(model, compressed_model_path, compress=compression_level)
print(f"Compressed model saved to {compressed_model_path}!")

# Print file sizes for comparison
original_size = os.path.getsize(original_model_path) / (1024 * 1024)
compressed_size = os.path.getsize(compressed_model_path) / (1024 * 1024)
print(f"Original size: {original_size:.2f} MB")
print(f"Compressed size: {compressed_size:.2f} MB")
