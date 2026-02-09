import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiService {
  // Base URL of your FastAPI backend
  static const String baseUrl = "https://nse-stock-app.onrender.com";

  // Health check with detailed error logging
  Future<bool> checkHealth() async {
    try {
      print('🔍 Attempting health check to: $baseUrl/');
      
      final response = await http.get(
        Uri.parse('$baseUrl/'),
        headers: {'Accept': 'application/json'},
      ).timeout(Duration(seconds: 10));
      
      print('✅ Health check response: ${response.statusCode}');
      print('📦 Response body: ${response.body}');
      
      return response.statusCode == 200;
    } on SocketException catch (e) {
      print('❌ SocketException (DNS/Network): $e');
      print('   Error code: ${e.osError?.errorCode}');
      print('   Message: ${e.osError?.message}');
      return false;
    } on HttpException catch (e) {
      print('❌ HttpException: $e');
      return false;
    } on FormatException catch (e) {
      print('❌ FormatException: $e');
      return false;
    } catch (e) {
      print('❌ Unknown error in health check: $e');
      return false;
    }
  }

  // Get all available stocks
  Future<List<Map<String, dynamic>>> getStocks() async {
    try {
      print('🔍 Fetching stocks from: $baseUrl/stocks');
      
      final response = await http.get(
        Uri.parse('$baseUrl/stocks'),
        headers: {'Accept': 'application/json'},
      ).timeout(Duration(seconds: 10));

      print('✅ Stocks response: ${response.statusCode}');

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        print('📦 Stocks data keys: ${data.keys}');
        return List<Map<String, dynamic>>.from(data['stocks']);
      } else {
        throw Exception('Failed to fetch stocks: ${response.statusCode}');
      }
    } on SocketException catch (e) {
      print('❌ SocketException in getStocks: $e');
      print('   Error code: ${e.osError?.errorCode}');
      rethrow;
    } catch (e) {
      print('❌ getStocks error: $e');
      rethrow;
    }
  }

  // Predict buy/sell/hold for a stock
  Future<Map<String, dynamic>> predictStock(String symbol) async {
    try {
      print('🔍 Predicting stock: $symbol');
      
      final response = await http.post(
        Uri.parse('$baseUrl/predict'),
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: jsonEncode({"symbol": symbol}),
      ).timeout(Duration(seconds: 15));

      print('✅ Predict response: ${response.statusCode}');

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Prediction failed: ${response.statusCode} - ${response.body}');
      }
    } on SocketException catch (e) {
      print('❌ SocketException in predictStock: $e');
      rethrow;
    } catch (e) {
      print('❌ predictStock error: $e');
      rethrow;
    }
  }

  // Get top stocks by criteria: gainers, losers, volume, buy_signals
  Future<List<Map<String, dynamic>>> getTopStocks({
    String criteria = "gainers",
    int limit = 10,
  }) async {
    try {
      final url = '$baseUrl/top-stocks?criteria=$criteria&limit=$limit';
      print('🔍 Fetching top stocks from: $url');
      
      final response = await http.get(
        Uri.parse(url),
        headers: {'Accept': 'application/json'},
      ).timeout(Duration(seconds: 15));

      print('✅ Top stocks response: ${response.statusCode}');

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        print('📦 Top stocks count: ${data['count']}');
        return List<Map<String, dynamic>>.from(data['stocks']);
      } else {
        print('❌ Failed status: ${response.statusCode}, Body: ${response.body}');
        throw Exception('Failed to fetch top stocks: ${response.statusCode}');
      }
    } on SocketException catch (e) {
      print('❌ SocketException in getTopStocks: $e');
      print('   Error code: ${e.osError?.errorCode}');
      print('   Message: ${e.osError?.message}');
      print('   Address: ${e.address}');
      print('   Port: ${e.port}');
      rethrow;
    } on HttpException catch (e) {
      print('❌ HttpException in getTopStocks: $e');
      rethrow;
    } catch (e) {
      print('❌ getTopStocks error: $e');
      rethrow;
    }
  }

  // Get overall market statistics
  Future<Map<String, dynamic>> getMarketOverview() async {
    try {
      print('🔍 Fetching market overview from: $baseUrl/market-overview');
      
      final response = await http.get(
        Uri.parse('$baseUrl/market-overview'),
        headers: {'Accept': 'application/json'},
      ).timeout(Duration(seconds: 10));

      print('✅ Market overview response: ${response.statusCode}');

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to fetch market overview: ${response.statusCode}');
      }
    } on SocketException catch (e) {
      print('❌ SocketException in getMarketOverview: $e');
      rethrow;
    } catch (e) {
      print('❌ getMarketOverview error: $e');
      rethrow;
    }
  }

  // Get trend analysis for a stock
  Future<Map<String, dynamic>> getTrends(String symbol, {int days = 30}) async {
    try {
      print('🔍 Fetching trends for: $symbol');
      
      final response = await http.get(
        Uri.parse('$baseUrl/trends/$symbol?days=$days'),
        headers: {'Accept': 'application/json'},
      ).timeout(Duration(seconds: 10));

      print('✅ Trends response: ${response.statusCode}');

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to fetch trends: ${response.statusCode}');
      }
    } on SocketException catch (e) {
      print('❌ SocketException in getTrends: $e');
      rethrow;
    } catch (e) {
      print('❌ getTrends error: $e');
      rethrow;
    }
  }

  // Get historical data for a stock
  Future<List<Map<String, dynamic>>> getHistory(String symbol, {int days = 30}) async {
    try {
      print('🔍 Fetching history for: $symbol');
      
      final response = await http.get(
        Uri.parse('$baseUrl/history/$symbol?days=$days'),
        headers: {'Accept': 'application/json'},
      ).timeout(Duration(seconds: 10));

      print('✅ History response: ${response.statusCode}');

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return List<Map<String, dynamic>>.from(data['data']);
      } else {
        throw Exception('Failed to fetch history: ${response.statusCode}');
      }
    } on SocketException catch (e) {
      print('❌ SocketException in getHistory: $e');
      rethrow;
    } catch (e) {
      print('❌ getHistory error: $e');
      rethrow;
    }
  }
}