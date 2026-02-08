import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  // Base URL of your FastAPI backend (replace with your PC's local IP)
  final String baseUrl = "http://192.168.1.181:8001";

  /// Health check
  Future<bool> checkHealth() async {
    final response = await http.get(Uri.parse('$baseUrl/'));
    return response.statusCode == 200;
  }

  /// Get all available stocks
  Future<List<Map<String, dynamic>>> getStocks() async {
    final response = await http.get(Uri.parse('$baseUrl/stocks'));
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return List<Map<String, dynamic>>.from(data['stocks_with_names']);
    } else {
      throw Exception('Failed to fetch stocks');
    }
  }

  /// Predict buy/sell/hold for a stock
  Future<Map<String, dynamic>> predictStock(String symbol) async {
    final response = await http.post(
      Uri.parse('$baseUrl/predict'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({"symbol": symbol}),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Prediction failed: ${response.body}');
    }
  }

  /// Get top stocks by criteria: gainers, losers, volume, buy_signals
  Future<List<Map<String, dynamic>>> getTopStocks({
    String criteria = "gainers",
    int limit = 10,
  }) async {
    final response = await http.get(
      Uri.parse('$baseUrl/top-stocks?criteria=$criteria&limit=$limit'),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return List<Map<String, dynamic>>.from(data['stocks']);
    } else {
      throw Exception('Failed to fetch top stocks');
    }
  }

  /// Get overall market statistics
  Future<Map<String, dynamic>> getMarketOverview() async {
    final response = await http.get(Uri.parse('$baseUrl/market-overview'));
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to fetch market overview');
    }
  }

  /// Get trend analysis for a stock
  Future<Map<String, dynamic>> getTrends(String symbol, {int days = 30}) async {
    final response = await http.get(
      Uri.parse('$baseUrl/trends/$symbol?days=$days'),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to fetch trends');
    }
  }

  /// Get historical data for a stock
  Future<List<Map<String, dynamic>>> getHistory(String symbol, {int days = 30}) async {
    final response = await http.get(
      Uri.parse('$baseUrl/history/$symbol?days=$days'),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return List<Map<String, dynamic>>.from(data['data']);
    } else {
      throw Exception('Failed to fetch history');
    }
  }
}
