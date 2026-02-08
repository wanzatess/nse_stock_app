import 'package:flutter/material.dart';
import '../services/api_service.dart';

class StockDetailScreen extends StatefulWidget {
  final String symbol;
  final String name;

  const StockDetailScreen({
    super.key,
    required this.symbol,
    required this.name,
  });

  @override
  State<StockDetailScreen> createState() => _StockDetailScreenState();
}

class _StockDetailScreenState extends State<StockDetailScreen> {
  final ApiService _apiService = ApiService();
  
  Map<String, dynamic>? predictionData;
  Map<String, dynamic>? trendsData;
  bool isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    setState(() => isLoading = true);
    
    try {
      final prediction = await _apiService.predictStock(widget.symbol);
      final trends = await _apiService.getTrends(widget.symbol, days: 30);

      setState(() {
        predictionData = prediction;
        trendsData = trends;
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

  Color _getPredictionColor(String prediction) {
    if (prediction.toLowerCase() == 'buy') return Colors.green;
    if (prediction.toLowerCase() == 'sell') return Colors.red;
    if (prediction.toLowerCase() == 'hold') return Colors.orange;
    return Colors.grey;
  }

  IconData _getPredictionIcon(String prediction) {
    if (prediction.toLowerCase() == 'buy') return Icons.trending_up;
    if (prediction.toLowerCase() == 'sell') return Icons.trending_down;
    if (prediction.toLowerCase() == 'hold') return Icons.trending_flat;
    return Icons.help_outline;
  }

  String _getTrendEmoji(String trend) {
    if (trend == 'upward') return '📈';
    if (trend == 'downward') return '📉';
    return '➡️';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(widget.symbol),
            Text(
              widget.name,
              style: const TextStyle(fontSize: 12),
            ),
          ],
        ),
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
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    if (predictionData != null) ...[
                      // Prediction Card
                      _buildPredictionCard(),
                      const SizedBox(height: 16),
                      
                      // Price Info Card
                      _buildPriceInfoCard(),
                      const SizedBox(height: 16),
                    ],
                    
                    if (trendsData != null) ...[
                      // Trends Card
                      _buildTrendsCard(),
                      const SizedBox(height: 16),
                      
                      // Technical Indicators
                      _buildTechnicalIndicatorsCard(),
                    ],
                  ],
                ),
              ),
      ),
    );
  }

  Widget _buildPredictionCard() {
    final prediction = predictionData!['prediction'];
    final color = _getPredictionColor(prediction);
    
    return Card(
      elevation: 4,
      color: color.withOpacity(0.1),
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            Icon(
              _getPredictionIcon(prediction),
              size: 64,
              color: color,
            ),
            const SizedBox(height: 16),
            const Text(
              'AI Prediction',
              style: TextStyle(
                fontSize: 16,
                color: Colors.grey,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              prediction.toString().toUpperCase(),
              style: TextStyle(
                fontSize: 36,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.blue,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                _getPredictionMessage(prediction),
                textAlign: TextAlign.center,
                style: const TextStyle(fontSize: 14),
              ),
            ),
          ],
        ),
      ),
    );
  }

  String _getPredictionMessage(String prediction) {
    switch (prediction.toLowerCase()) {
      case 'buy':
        return 'Strong buy signal detected. Consider adding to your portfolio.';
      case 'sell':
        return 'Sell signal detected. Consider reducing your position.';
      case 'hold':
        return 'Hold your current position. Wait for better entry/exit points.';
      default:
        return 'Unable to determine clear signal.';
    }
  }

  Widget _buildPriceInfoCard() {
    final currentPrice = predictionData!['current_price'];
    final previousPrice = predictionData!['previous_price'];
    final change = predictionData!['change'];
    final changePercent = predictionData!['change_percent'];
    final dayLow = predictionData!['day_low'];
    final dayHigh = predictionData!['day_high'];
    final low12m = predictionData!['12m_low'];
    final high12m = predictionData!['12m_high'];
    final volume = predictionData!['volume'];
    
    final isPositive = change >= 0;
    
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Current Price',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'KSh ${currentPrice.toStringAsFixed(2)}',
                      style: const TextStyle(
                        fontSize: 32,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  decoration: BoxDecoration(
                    color: isPositive ? Colors.green : Colors.red,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      Text(
                        '${isPositive ? '+' : ''}${change.toStringAsFixed(2)}',
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Text(
                        '${isPositive ? '+' : ''}${changePercent.toStringAsFixed(2)}%',
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 14,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 24),
            const Divider(),
            const SizedBox(height: 16),
            
            _buildInfoRow('Previous Close', 'KSh ${previousPrice.toStringAsFixed(2)}'),
            const SizedBox(height: 12),
            _buildInfoRow('Day Range', 'KSh ${dayLow.toStringAsFixed(2)} - KSh ${dayHigh.toStringAsFixed(2)}'),
            const SizedBox(height: 12),
            _buildInfoRow('52 Week Range', 'KSh ${low12m.toStringAsFixed(2)} - KSh ${high12m.toStringAsFixed(2)}'),
            const SizedBox(height: 12),
            _buildInfoRow('Volume', _formatVolume(volume)),
          ],
        ),
      ),
    );
  }

  Widget _buildTrendsCard() {
    final trend = trendsData!['trend'];
    final highestPrice = trendsData!['highest_price'];
    final lowestPrice = trendsData!['lowest_price'];
    final avgPrice = trendsData!['average_price'];
    final priceChange = trendsData!['price_change'];
    final periodDays = trendsData!['period_days'];
    
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.show_chart),
                const SizedBox(width: 8),
                const Text(
                  '30-Day Trend Analysis',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            
            // Trend Indicator
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: _getTrendColor(trend).withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(
                  color: _getTrendColor(trend).withOpacity(0.3),
                  width: 2,
                ),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    _getTrendEmoji(trend),
                    style: const TextStyle(fontSize: 32),
                  ),
                  const SizedBox(width: 12),
                  Text(
                    '${trend.toUpperCase()} TREND',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: _getTrendColor(trend),
                    ),
                  ),
                ],
              ),
            ),
            
            const SizedBox(height: 20),
            const Divider(),
            const SizedBox(height: 16),
            
            _buildInfoRow('Period', '$periodDays days'),
            const SizedBox(height: 12),
            _buildInfoRow('Highest', 'KSh ${highestPrice.toStringAsFixed(2)}'),
            const SizedBox(height: 12),
            _buildInfoRow('Lowest', 'KSh ${lowestPrice.toStringAsFixed(2)}'),
            const SizedBox(height: 12),
            _buildInfoRow('Average', 'KSh ${avgPrice.toStringAsFixed(2)}'),
            const SizedBox(height: 12),
            _buildInfoRow(
              'Price Change',
              '${priceChange >= 0 ? '+' : ''}KSh ${priceChange.toStringAsFixed(2)}',
              valueColor: priceChange >= 0 ? Colors.green : Colors.red,
            ),
          ],
        ),
      ),
    );
  }

  Color _getTrendColor(String trend) {
    if (trend == 'upward') return Colors.green;
    if (trend == 'downward') return Colors.red;
    return Colors.orange;
  }

  Widget _buildTechnicalIndicatorsCard() {
    if (predictionData?['features_used'] == null) return const SizedBox();
    
    final features = predictionData!['features_used'];
    final ma5 = features['ma_5'];
    final ma10 = features['ma_10'];
    final volatility = features['daily_volatility'];
    final pctFrom12mLow = features['pct_from_12m_low'];
    final pctFrom12mHigh = features['pct_from_12m_high'];
    
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.analytics),
                const SizedBox(width: 8),
                const Text(
                  'Technical Indicators',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            
            _buildInfoRow('5-Day MA', 'KSh ${ma5.toStringAsFixed(2)}'),
            const SizedBox(height: 12),
            _buildInfoRow('10-Day MA', 'KSh ${ma10.toStringAsFixed(2)}'),
            const SizedBox(height: 12),
            _buildInfoRow('Daily Volatility', volatility.toStringAsFixed(2)),
            const SizedBox(height: 12),
            _buildInfoRow(
              'From 52W Low',
              '${pctFrom12mLow >= 0 ? '+' : ''}${pctFrom12mLow.toStringAsFixed(2)}%',
              valueColor: pctFrom12mLow >= 0 ? Colors.green : Colors.red,
            ),
            const SizedBox(height: 12),
            _buildInfoRow(
              'From 52W High',
              '${pctFrom12mHigh >= 0 ? '+' : ''}${pctFrom12mHigh.toStringAsFixed(2)}%',
              valueColor: pctFrom12mHigh >= 0 ? Colors.green : Colors.red,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoRow(String label, String value, {Color? valueColor}) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: TextStyle(
            color: Colors.grey.shade700,
            fontSize: 14,
          ),
        ),
        Text(
          value,
          style: TextStyle(
            color: valueColor,
            fontSize: 14,
            fontWeight: FontWeight.w600,
          ),
        ),
      ],
    );
  }

  String _formatVolume(int volume) {
    if (volume >= 10000000) {
      return '${(volume / 10000000).toStringAsFixed(2)} Cr';
    } else if (volume >= 100000) {
      return '${(volume / 100000).toStringAsFixed(2)} L';
    } else if (volume >= 1000) {
      return '${(volume / 1000).toStringAsFixed(2)} K';
    }
    return volume.toString();
  }
}
