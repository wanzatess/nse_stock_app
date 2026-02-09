import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  // Base URL of your FastAPI backend
  static const String baseUrl = "https://nse-stock-app.onrender.com";

  // Health check
  Future<bool> checkHealth() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/'),
        headers: {'Accept': 'application/json'},
      ).timeout(Duration(seconds: 10));
      
      return response.statusCode == 200;
    } catch (e) {
      print('Health check failed: $e');
      return false;
    }
  }

  // Get all available stocks
  Future<List<Map<String, dynamic>>> getStocks() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/stocks'),
        headers: {'Accept': 'application/json'},
      ).timeout(Duration(seconds: 10));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return List<Map<String, dynamic>>.from(data['stocks_with_names']);
      } else {
        throw Exception('Failed to fetch stocks: ${response.statusCode}');
      }
    } catch (e) {
      print('getStocks error: $e');
      rethrow;
    }
  }

  // Predict buy/sell/hold for a stock
  Future<Map<String, dynamic>> predictStock(String symbol) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/predict'),
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: jsonEncode({"symbol": symbol}),
      ).timeout(Duration(seconds: 15));

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Prediction failed: ${response.statusCode} - ${response.body}');
      }
    } catch (e) {
      print('predictStock error: $e');
      rethrow;
    }
  }

  // Get top stocks by criteria: gainers, losers, volume, buy_signals
  Future<List<Map<String, dynamic>>> getTopStocks({
    String criteria = "gainers",
    int limit = 10,
  }) async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/top-stocks?criteria=$criteria&limit=$limit'),
        headers: {'Accept': 'application/json'},
      ).timeout(Duration(seconds: 10));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return List<Map<String, dynamic>>.from(data['stocks']);
      } else {
        throw Exception('Failed to fetch top stocks: ${response.statusCode}');
      }
    } catch (e) {
      print('getTopStocks error: $e');
      rethrow;
    }
  }

  // Get overall market statistics
  Future<Map<String, dynamic>> getMarketOverview() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/market-overview'),
        headers: {'Accept': 'application/json'},
      ).timeout(Duration(seconds: 10));

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to fetch market overview: ${response.statusCode}');
      }
    } catch (e) {
      print('getMarketOverview error: $e');
      rethrow;
    }
  }

  // Get trend analysis for a stock
  Future<Map<String, dynamic>> getTrends(String symbol, {int days = 30}) async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/trends/$symbol?days=$days'),
        headers: {'Accept': 'application/json'},
      ).timeout(Duration(seconds: 10));

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to fetch trends: ${response.statusCode}');
      }
    } catch (e) {
      print('getTrends error: $e');
      rethrow;
    }
  }

  // Get historical data for a stock
  Future<List<Map<String, dynamic>>> getHistory(String symbol, {int days = 30}) async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/history/$symbol?days=$days'),
        headers: {'Accept': 'application/json'},
      ).timeout(Duration(seconds: 10));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return List<Map<String, dynamic>>.from(data['data']);
      } else {
        throw Exception('Failed to fetch history: ${response.statusCode}');
      }
    } catch (e) {
      print('getHistory error: $e');
      rethrow;
    }
  }
}