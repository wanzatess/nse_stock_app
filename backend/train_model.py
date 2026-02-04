import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os

print("Script started...")  # <-- you should see this

# Paths
data_path = "../data/processed/NSE_20_stocks_2013_2025_features_target.csv"
model_folder = "models"
os.makedirs(model_folder, exist_ok=True)
model_path = os.path.join(model_folder, "stock_model.pkl")

# Load dataset
df = pd.read_csv(data_path)
print("Data loaded. Rows:", len(df))  # <-- check if data loads

# Features & target
feature_cols = ['day_price', 'ma_5', 'ma_10', 'pct_from_12m_low', 'pct_from_12m_high', 'daily_return', 'daily_volatility']
X = df[feature_cols].fillna(0)
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model trained.")  # <-- check if training completes

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model saved as: {model_path}")
