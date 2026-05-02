# 📈 NSE Stock App

A full-stack stock tracking and prediction platform for the **Nairobi Securities Exchange (NSE)**.  
Combines real-time market data, machine learning price predictions, a Flutter mobile app, and a web dashboard — all deployed on Render.

---

## 🏗️ Architecture

**nse_stock_app**
- backend       # FastAPI REST API + ML prediction engine
- mobile        # Flutter mobile app (Dart)
- website       # Web dashboard (JavaScript)
- notebooks     # Jupyter notebooks for data analysis & model training
- data          # NSE market datasets
## ✨ Features

- 📊 **Live stock data** — fetches NSE prices via `yfinance` and web scraping (`BeautifulSoup`)
- 🤖 **ML price predictions** — trained models using `scikit-learn`, served via FastAPI
- 📱 **Flutter mobile app** — cross-platform iOS/Android stock tracker
- 🌐 **Web dashboard** — browser-based interface for market overview
- ☁️ **Deployed on Render** — backend API and static site hosted in production

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI, Uvicorn, Python |
| ML / Data | scikit-learn, pandas, numpy, yfinance |
| Data Collection | BeautifulSoup4, lxml, requests |
| Mobile | Flutter, Dart |
| Web | JavaScript |
| Deployment | Render |

## 🚀 Setup

### Backend (FastAPI)
```bash
git clone https://github.com/wanzatess/nse_stock_app.git
cd nse_stock_app

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Start the API server
uvicorn backend.app:app --reload
# API available at http://localhost:8000
```

### Mobile App (Flutter)
```bash
cd mobile
flutter pub get
flutter run
```

### Web Dashboard
```bash
cd website
npm install
npm run build
```

### Jupyter Notebooks (Data Analysis & Model Training)
```bash
pip install -r requirements_predictions.txt
jupyter notebook notebooks/
```

## ☁️ Deployment

The app is configured for one-click deployment on Render via `render.yaml`:
- **Backend**: Python web service running FastAPI
- **Website**: Static site built from the `website/` directory

## 📊 Data

NSE market data is stored in the `data/` directory and fetched live using `yfinance` and custom NSE scrapers.

---

*Built for Kenyan investors 🇰🇪*
