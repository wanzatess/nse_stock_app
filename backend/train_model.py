import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib  # Changed from pickle to joblib for consistency
import os

print("🚀 Script started...")

# Paths
data_path = "../data/processed/NSE_20_stocks_2022_2026.csv"
model_folder = "models"
os.makedirs(model_folder, exist_ok=True)
model_path = os.path.join(model_folder, "stock_model.pkl")

# Load dataset
print("📂 Loading data...")
df = pd.read_csv(data_path)
print(f"✅ Data loaded. Rows: {len(df)}")

# Features & target
feature_cols = ['day_price', 'ma_5', 'ma_10', 'pct_from_12m_low', 'pct_from_12m_high', 'daily_return', 'daily_volatility']

# Drop rows where target is NaN
df = df.dropna(subset=['target'])

X = df[feature_cols].fillna(0)
y = df['target']

print(f"📊 Features: {X.shape[0]} rows, {X.shape[1]} columns")
print(f"🎯 Target distribution:\n{y.value_counts()}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n🔀 Train set: {len(X_train)} | Test set: {len(X_test)}")

# Train model
print("\n🤖 Training RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# Evaluate
print("\n📈 Evaluating model...")
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate accuracy
accuracy = model.score(X_test, y_test)
print(f"\n🎯 Test Accuracy: {accuracy:.2%}")

# Save model using joblib (same as app.py uses for loading)
print(f"\n💾 Saving model to: {model_path}")
joblib.dump(model, model_path)
print("✅ Model saved successfully!")

print("\n🎉 All done! Now upload this model to Google Drive and update your app.")