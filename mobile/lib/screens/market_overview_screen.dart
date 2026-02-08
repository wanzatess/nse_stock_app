import 'package:flutter/material.dart';
import '../services/api_service.dart';

class MarketOverviewScreen extends StatefulWidget {
  const MarketOverviewScreen({super.key});

  @override
  State<MarketOverviewScreen> createState() => _MarketOverviewScreenState();
}

class _MarketOverviewScreenState extends State<MarketOverviewScreen> {
  final ApiService _apiService = ApiService();
  Map<String, dynamic>? marketData;
  bool isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadMarketData();
  }

  Future<void> _loadMarketData() async {
    setState(() => isLoading = true);
    
    try {
      final data = await _apiService.getMarketOverview();
      setState(() {
        marketData = data;
        isLoading = false;
      });
    } catch (e) {
      setState(() => isLoading = false);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error loading market data: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Market Overview'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadMarketData,
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: _loadMarketData,
        child: isLoading
            ? const Center(child: CircularProgressIndicator())
            : marketData == null
                ? Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.error_outline, size: 64, color: Colors.grey.shade400),
                        const SizedBox(height: 16),
                        const Text('Failed to load market data'),
                        const SizedBox(height: 16),
                        ElevatedButton(
                          onPressed: _loadMarketData,
                          child: const Text('Retry'),
                        ),
                      ],
                    ),
                  )
                : SingleChildScrollView(
                    physics: const AlwaysScrollableScrollPhysics(),
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        _buildMarketSummaryCard(),
                        const SizedBox(height: 20),
                        _buildMarketBreakdownCard(),
                        const SizedBox(height: 20),
                        _buildMarketStatsCard(),
                      ],
                    ),
                  ),
      ),
    );
  }

  Widget _buildMarketSummaryCard() {
    final totalStocks = marketData!['total_stocks'] ?? 0;
    final avgChange = marketData!['average_change'] ?? 0.0;
    final isPositive = avgChange >= 0;
    
    return Card(
      elevation: 4,
      child: Container(
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(12),
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: isPositive
                ? [Colors.green.shade400, Colors.green.shade600]
                : [Colors.red.shade400, Colors.red.shade600],
          ),
        ),
        child: Column(
          children: [
            const Text(
              'NSE 20 Market',
              style: TextStyle(
                fontSize: 18,
                color: Colors.white70,
                fontWeight: FontWeight.w500,
              ),
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Icon(
                  isPositive ? Icons.trending_up : Icons.trending_down,
                  size: 48,
                  color: Colors.white,
                ),
                const SizedBox(width: 12),
                Text(
                  '${isPositive ? '+' : ''}${avgChange.toStringAsFixed(2)}%',
                  style: const TextStyle(
                    fontSize: 48,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              'Average Change',
              style: TextStyle(
                fontSize: 14,
                color: Colors.white.withOpacity(0.9),
              ),
            ),
            const SizedBox(height: 20),
            Divider(color: Colors.white.withOpacity(0.3)),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildSummaryItem('Total Stocks', totalStocks.toString()),
                _buildSummaryItem('Last Updated', marketData!['last_updated'] ?? 'N/A'),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSummaryItem(String label, String value) {
    return Column(
      children: [
        Text(
          value,
          style: const TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          label,
          style: TextStyle(
            fontSize: 12,
            color: Colors.white.withOpacity(0.8),
          ),
        ),
      ],
    );
  }

  Widget _buildMarketBreakdownCard() {
    final gainers = marketData!['gainers'] ?? 0;
    final losers = marketData!['losers'] ?? 0;
    final unchanged = marketData!['unchanged'] ?? 0;
    final total = gainers + losers + unchanged;
    
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Market Breakdown',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 20),
            
            // Gainers
            _buildProgressBar(
              'Gainers',
              gainers,
              total,
              Colors.green,
              Icons.trending_up,
            ),
            const SizedBox(height: 16),
            
            // Losers
            _buildProgressBar(
              'Losers',
              losers,
              total,
              Colors.red,
              Icons.trending_down,
            ),
            const SizedBox(height: 16),
            
            // Unchanged
            _buildProgressBar(
              'Unchanged',
              unchanged,
              total,
              Colors.orange,
              Icons.trending_flat,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildProgressBar(String label, int value, int total, Color color, IconData icon) {
    final percentage = total > 0 ? (value / total * 100) : 0.0;
    
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Row(
              children: [
                Icon(icon, size: 20, color: color),
                const SizedBox(width: 8),
                Text(
                  label,
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
            Text(
              '$value (${percentage.toStringAsFixed(1)}%)',
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: LinearProgressIndicator(
            value: percentage / 100,
            minHeight: 8,
            backgroundColor: color.withOpacity(0.2),
            valueColor: AlwaysStoppedAnimation<Color>(color),
          ),
        ),
      ],
    );
  }

  Widget _buildMarketStatsCard() {
    final totalVolume = marketData!['total_volume'] ?? 0;
    
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Market Statistics',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 20),
            _buildStatRow('Total Volume', _formatVolume(totalVolume), Icons.bar_chart),
            const Divider(height: 24),
            _buildStatRow('Market Sentiment', _getMarketSentiment(), Icons.sentiment_satisfied_alt),
          ],
        ),
      ),
    );
  }

  Widget _buildStatRow(String label, String value, IconData icon) {
    return Row(
      children: [
        Icon(icon, color: Theme.of(context).colorScheme.primary),
        const SizedBox(width: 12),
        Expanded(
          child: Text(
            label,
            style: const TextStyle(fontSize: 14),
          ),
        ),
        Text(
          value,
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }

  String _formatVolume(int volume) {
    if (volume >= 1000000000) {
      return '${(volume / 1000000000).toStringAsFixed(2)}B';
    } else if (volume >= 10000000) {
      return '${(volume / 10000000).toStringAsFixed(2)}Cr';
    } else if (volume >= 100000) {
      return '${(volume / 100000).toStringAsFixed(2)}L';
    } else if (volume >= 1000) {
      return '${(volume / 1000).toStringAsFixed(2)}K';
    }
    return volume.toString();
  }

  String _getMarketSentiment() {
    final avgChange = marketData!['average_change'] ?? 0.0;
    
    if (avgChange > 1.0) return 'Very Bullish 🚀';
    if (avgChange > 0.5) return 'Bullish 📈';
    if (avgChange > 0) return 'Slightly Positive ↗️';
    if (avgChange == 0) return 'Neutral ➡️';
    if (avgChange > -0.5) return 'Slightly Negative ↘️';
    if (avgChange > -1.0) return 'Bearish 📉';
    return 'Very Bearish 💥';
  }
}
