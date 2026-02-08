import 'package:flutter/material.dart';
import '../services/api_service.dart';
import 'stock_detail_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ApiService _apiService = ApiService();
  Map<String, dynamic>? marketOverview;
  List<dynamic> topGainers = [];
  List<dynamic> buySignals = [];
  bool isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    setState(() => isLoading = true);
    
    try {
      final overview = await _apiService.getMarketOverview();
      final gainers = await _apiService.getTopStocks(criteria: 'gainers', limit: 5);
      final buys = await _apiService.getTopStocks(criteria: 'buy_signals', limit: 5);

      setState(() {
        marketOverview = overview;
        topGainers = gainers;
        buySignals = buys;
        isLoading = false;
      });
    } catch (e) {
      setState(() => isLoading = false);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error loading data: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('NSE Stock Predictor'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadData,
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: _loadData,
        child: isLoading
            ? const Center(child: CircularProgressIndicator())
            : SingleChildScrollView(
                physics: const AlwaysScrollableScrollPhysics(),
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Market Overview Card
                    _buildMarketOverviewCard(),
                    
                    const SizedBox(height: 24),
                    
                    // Top Gainers Section
                    _buildSectionHeader('Top Gainers', Icons.trending_up),
                    const SizedBox(height: 12),
                    _buildStockList(topGainers),
                    
                    const SizedBox(height: 24),
                    
                    // Buy Signals Section
                    _buildSectionHeader('Buy Signals', Icons.shopping_cart),
                    const SizedBox(height: 12),
                    _buildStockList(buySignals, isBuySignal: true),
                  ],
                ),
              ),
      ),
    );
  }

  Widget _buildMarketOverviewCard() {
    if (marketOverview == null) return const SizedBox();
    
    final gainers = marketOverview!['gainers'] ?? 0;
    final losers = marketOverview!['losers'] ?? 0;
    final unchanged = marketOverview!['unchanged'] ?? 0;
    final avgChange = marketOverview!['average_change'] ?? 0.0;
    
    return Card(
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'Market Overview',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: avgChange >= 0 ? Colors.green : Colors.red,
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    '${avgChange >= 0 ? '+' : ''}${avgChange.toStringAsFixed(2)}%',
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildMarketStat('Gainers', gainers.toString(), Colors.green),
                _buildMarketStat('Losers', losers.toString(), Colors.red),
                _buildMarketStat('Unchanged', unchanged.toString(), Colors.orange),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMarketStat(String label, String value, Color color) {
    return Column(
      children: [
        Text(
          value,
          style: TextStyle(
            fontSize: 28,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          label,
          style: TextStyle(
            fontSize: 12,
            color: Colors.grey.shade600,
          ),
        ),
      ],
    );
  }

  Widget _buildSectionHeader(String title, IconData icon) {
    return Row(
      children: [
        Icon(icon, size: 24),
        const SizedBox(width: 8),
        Text(
          title,
          style: const TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }

  Widget _buildStockList(List<dynamic> stocks, {bool isBuySignal = false}) {
    if (stocks.isEmpty) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Center(
            child: Text(
              'No ${isBuySignal ? 'buy signals' : 'data'} available',
              style: TextStyle(color: Colors.grey.shade600),
            ),
          ),
        ),
      );
    }
    
    return Column(
      children: stocks.map((stock) => _buildStockCard(stock)).toList(),
    );
  }

  Widget _buildStockCard(Map<String, dynamic> stock) {
    final change = stock['change'] ?? 0.0;
    final changePercent = stock['change_percent'] ?? 0.0;
    final isPositive = change >= 0;
    
    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: InkWell(
        onTap: () {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => StockDetailScreen(
                symbol: stock['symbol'],
                name: stock['name'],
              ),
            ),
          );
        },
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              // Stock Info
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      stock['symbol'],
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      stock['name'],
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey.shade600,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ],
                ),
              ),
              
              // Price Info
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    'KSh ${stock['current_price'].toStringAsFixed(2)}',
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: isPositive ? Colors.green.withOpacity(0.1) : Colors.red.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text(
                      '${isPositive ? '+' : ''}${changePercent.toStringAsFixed(2)}%',
                      style: TextStyle(
                        fontSize: 12,
                        color: isPositive ? Colors.green : Colors.red,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
