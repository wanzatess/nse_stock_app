# NSE Stock Predictor App

A beautiful Flutter app for predicting NSE (Nairobi Securities Exchange) stock movements using AI.

## Features

✅ **AI-Powered Predictions** - Get Buy/Sell/Hold signals for NSE stocks
✅ **Real-time Prices** - View current stock prices in Kenyan Shillings (KSh)
✅ **Market Overview** - See overall market statistics and trends
✅ **Top Stocks** - Find top gainers, losers, most traded, and buy signals
✅ **Stock Search** - Search stocks by symbol or company name
✅ **Detailed Analysis** - View 30-day trends and technical indicators
✅ **Beautiful UI** - Material Design 3 with dark mode support

## Screenshots

[Add screenshots here]

## Setup Instructions

### Backend Setup

1. Make sure your backend is running on port 8001:
```bash
cd backend
python app.py
```

2. Update the IP address in `lib/services/api_service.dart`:
```dart
static const String baseUrl = 'http://YOUR_IP:8001';
```

### Flutter Setup

1. Install dependencies:
```bash
flutter pub get
```

2. Run the app:
```bash
flutter run
```

## App Structure

```
lib/
├── main.dart                          # App entry point with navigation
├── services/
│   └── api_service.dart              # API calls to backend
└── screens/
    ├── home_screen.dart              # Market overview and quick access
    ├── search_screen.dart            # Search stocks
    ├── top_stocks_screen.dart        # Top gainers/losers/volume
    ├── market_overview_screen.dart   # Market statistics
    └── stock_detail_screen.dart      # Individual stock analysis
```

## API Endpoints Used

- `GET /stocks` - Get all available stocks
- `POST /predict` - Get prediction for a stock
- `GET /top-stocks?criteria={gainers|losers|volume|buy_signals}` - Get top stocks
- `GET /market-overview` - Get market statistics
- `GET /trends/{symbol}` - Get 30-day trend analysis
- `GET /history/{symbol}` - Get historical data

## Currency

All prices are displayed in Kenyan Shillings (KSh).

## Requirements

- Flutter SDK >=3.0.0
- Dart SDK
- iOS/Android device or emulator
- Backend API running

## Contributing

Feel free to submit issues and pull requests!

## License

MIT License