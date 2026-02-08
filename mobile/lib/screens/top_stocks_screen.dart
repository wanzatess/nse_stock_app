import 'package:flutter/material.dart';
import '../services/api_service.dart';
import 'stock_detail_screen.dart';

class TopStocksScreen extends StatefulWidget {
  const TopStocksScreen({super.key});

  @override
  State<TopStocksScreen> createState() => _TopStocksScreenState();
}

class _TopStocksScreenState extends State<TopStocksScreen> with SingleTickerProviderStateMixin {
  final ApiService _apiService = ApiService();
  late TabController _tabController;
  
  Map<String, List<dynamic>> stocksData = {
    'gainers': [],
    'losers': [],
    'volume': [],
    'buy_signals': [],
  };
  
  Map<String, bool> isLoading = {
    'gainers': true,
    'losers': true,
    'volume': true,
    'buy_signals': true,
  };

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 4, vsync: this);
    _loadAllData();
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  Future<void> _loadAllData() async {
    _loadData('gainers');
    _loadData('losers');
    _loadData('volume');
    _loadData('buy_signals');
  }

  Future<void> _loadData(String criteria) async {
    setState(() => isLoading[criteria] = true);
    
    try {
      final result = await _apiService.getTopStocks(criteria: criteria, limit: 20);
      setState(() {
        stocksData[criteria] = result;
        isLoading[criteria] = false;
      });
    } catch (e) {
      setState(() => isLoading[criteria] = false);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error loading $criteria: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Top Stocks'),
        bottom: TabBar(
          controller: _tabController,
          isScrollable: true,
          tabs: const [
            Tab(icon: Icon(Icons.trending_up), text: 'Gainers'),
            Tab(icon: Icon(Icons.trending_down), text: 'Losers'),
            Tab(icon: Icon(Icons.bar_chart), text: 'Volume'),
            Tab(icon: Icon(Icons.recommend), text: 'Buy Signals'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          _buildStockList('gainers', Colors.green),
          _buildStockList('losers', Colors.red),
          _buildStockList('volume', Colors.blue),
          _buildStockList('buy_signals', Colors.orange),
        ],
      ),
    );
  }

  Widget _buildStockList(String criteria, Color accentColor) {
    if (isLoading[criteria] == true) {
      return const Center(child: CircularProgressIndicator());
    }
    
    final stocks = stocksData[criteria] ?? [];
    
    if (stocks.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.inbox, size: 64, color: Colors.grey.shade400),
            const SizedBox(height: 16),
            Text(
              'No data available',
              style: TextStyle(fontSize: 18, color: Colors.grey.shade600),
            ),
          ],
        ),
      );
    }
    
    return RefreshIndicator(
      onRefresh: () => _loadData(criteria),
      child: ListView.builder(
        padding: const EdgeInsets.all(16),
        itemCount: stocks.length,
        itemBuilder: (context, index) {
          final stock = stocks[index];
          final rank = index + 1;
          
          return _buildStockCard(stock, rank, accentColor, criteria);
        },
      ),
    );
  }

  Widget _buildStockCard(Map<String, dynamic> stock, int rank, Color accentColor, String criteria) {
    final change = stock['change'] ?? 0.0;
    final changePercent = stock['change_percent'] ?? 0.0;
    final isPositive = change >= 0;
    
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
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
              // Rank Badge
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  color: accentColor.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Center(
                  child: Text(
                    '#$rank',
                    style: TextStyle(
                      color: accentColor,
                      fontWeight: FontWeight.bold,
                      fontSize: 14,
                    ),
                  ),
                ),
              ),
              
              const SizedBox(width: 16),
              
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
                    if (criteria == 'volume') ...[
                      const SizedBox(height: 4),
                      Text(
                        'Vol: ${_formatVolume(stock['volume'])}',
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.grey.shade700,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
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
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          isPositive ? Icons.arrow_upward : Icons.arrow_downward,
                          size: 12,
                          color: isPositive ? Colors.green : Colors.red,
                        ),
                        const SizedBox(width: 4),
                        Text(
                          '${isPositive ? '+' : ''}${changePercent.toStringAsFixed(2)}%',
                          style: TextStyle(
                            fontSize: 12,
                            color: isPositive ? Colors.green : Colors.red,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
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

  String _formatVolume(int volume) {
    if (volume >= 10000000) {
      return '${(volume / 10000000).toStringAsFixed(2)}Cr';
    } else if (volume >= 100000) {
      return '${(volume / 100000).toStringAsFixed(2)}L';
    } else if (volume >= 1000) {
      return '${(volume / 1000).toStringAsFixed(2)}K';
    }
    return volume.toString();
  }
}
