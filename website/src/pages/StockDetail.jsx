import React, { useEffect, useState, useCallback } from "react";
import { predictStock, getTrends } from "../api/apiService";
import "../styles/StockDetail.css";

export default function StockDetail({ symbol, name, onBack }) {
  const [predictionData, setPredictionData] = useState(null);
  const [trendsData, setTrendsData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [predictionError, setPredictionError] = useState(null);
  const [trendsError, setTrendsError] = useState(null);

  const loadData = useCallback(async () => {
    if (!symbol) return;
    setLoading(true);
    setPredictionError(null);
    setTrendsError(null);

    // Fetch prediction and trends independently so failure of one doesn't block the other
    try {
      const predRes = await predictStock(symbol).catch((e) => {
        // capture prediction-specific error
        const msg = e?.response?.data?.detail || e?.message || "Model not available. Please try again later.";
        setPredictionError(msg);
        return null;
      });
      setPredictionData(predRes ? predRes.data : null);
    } catch (e) {
      console.error("Prediction error", e);
      setPredictionError(e?.message || "Prediction failed");
      setPredictionData(null);
    }

    try {
      const trendsRes = await getTrends(symbol, 30).catch((e) => {
        const msg = e?.response?.data?.detail || e?.message || "Failed to load trends";
        setTrendsError(msg);
        return null;
      });
      setTrendsData(trendsRes ? trendsRes.data : null);
    } catch (e) {
      console.error("Trends error", e);
      setTrendsError(e?.message || "Failed to load trends");
      setTrendsData(null);
    }

    setLoading(false);
  }, [symbol]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  function mapPrediction(pred) {
    // Backend returns prediction object under `prediction` key. Try to determine a human label.
    if (!pred) return { label: "N/A", color: "#9e9e9e", icon: "❓" };

    // Try nested shape: { prediction: { prediction: <num>, confidence: <float> } }
    let raw = pred.prediction ?? pred;
    if (raw && typeof raw === "object" && raw.prediction !== undefined) raw = raw.prediction;

    // If numeric: >0 buy, <0 sell, 0 hold
    if (typeof raw === "number") {
      if (raw > 0) return { label: "BUY", color: "#1b5e20", icon: "▲" };
      if (raw < 0) return { label: "SELL", color: "#b00020", icon: "▼" };
      return { label: "HOLD", color: "#ff9800", icon: "▬" };
    }

    // If string
    const s = String(raw).toLowerCase();
    if (s.includes("buy")) return { label: "BUY", color: "#1b5e20", icon: "▲" };
    if (s.includes("sell")) return { label: "SELL", color: "#b00020", icon: "▼" };
    if (s.includes("hold")) return { label: "HOLD", color: "#ff9800", icon: "▬" };

    return { label: String(raw).toUpperCase(), color: "#9e9e9e", icon: "❓" };
  }

  function formatVolume(v) {
    const n = Number(v) || 0;
    if (n >= 1000000000) return `${(n / 1000000000).toFixed(2)}B`;
    if (n >= 10000000) return `${(n / 10000000).toFixed(2)}Cr`;
    if (n >= 100000) return `${(n / 100000).toFixed(2)}L`;
    if (n >= 1000) return `${(n / 1000).toFixed(2)}K`;
    return String(n);
  }

  if (!symbol) return <div className="stock-detail-empty">No symbol provided</div>;

  if (loading) return <div className="stock-detail-loading">Loading...</div>;
  // If neither prediction nor trends loaded, show combined errors (if any)
  if (!predictionData && !trendsData) {
    const msgs = [predictionError, trendsError].filter(Boolean);
    return (
      <div className="stock-detail-error">{msgs.length ? msgs.join(' — ') : 'No data available'}</div>
    );
  }

  const predMeta = mapPrediction(predictionData);
  const currentPrice = predictionData?.current_price ?? predictionData?.day_price ?? 0;
  const previousPrice = predictionData?.previous_price ?? 0;
  const change = predictionData?.change ?? 0;
  const changePercent = predictionData?.change_percent ?? 0;

  return (
    <div className="stock-detail">
      <div className="detail-header">
        <div>
          <div className="symbol">{symbol}</div>
          <div className="company">{name || predictionData?.name}</div>
        </div>
        <div className="header-actions">
          <button className="btn" onClick={loadData} title="Refresh">🔄</button>
          {onBack && <button className="btn" onClick={onBack}>←</button>}
        </div>
      </div>

      {/* Prediction Card */}
      {predictionData && (
        <div className="card prediction-card" style={{ borderColor: predMeta.color }}>
          <div className="prediction-icon" style={{ color: predMeta.color }}>{predMeta.icon}</div>
          <div className="prediction-body">
            <div className="prediction-label">AI Prediction</div>
            <div className="prediction-value" style={{ color: predMeta.color }}>{predMeta.label}</div>
            <div className="prediction-msg">
              {predMeta.label === 'BUY' && 'Strong buy signal detected. Consider adding to your portfolio.'}
              {predMeta.label === 'SELL' && 'Sell signal detected. Consider reducing your position.'}
              {predMeta.label === 'HOLD' && 'Hold your current position. Wait for better entry/exit points.'}
              {['N/A'].includes(predMeta.label) && 'No clear signal available.'}
            </div>
          </div>
        </div>
      )}

      {/* Price Info Card */}
      {predictionData && (
        <div className="card price-card">
          <div className="price-row">
            <div>
              <div className="label">Current Price</div>
              <div className="price">KSh {Number(currentPrice).toFixed(2)}</div>
            </div>
            <div className="price-badge" style={{ background: change >= 0 ? 'rgba(76,175,80,0.12)' : 'rgba(244,67,54,0.12)', color: change >=0 ? '#2e7d32' : '#b00020' }}>
              <div className="price-change">{change >= 0 ? '+' : ''}{Number(change).toFixed(2)}</div>
              <div className="price-change-pct">{change >= 0 ? '+' : ''}{Number(changePercent).toFixed(2)}%</div>
            </div>
          </div>

          <div className="info-rows">
            <div className="info-row"><span>Previous Close</span><span>KSh {Number(previousPrice).toFixed(2)}</span></div>
            <div className="info-row"><span>Day Range</span><span>KSh {Number(predictionData?.day_low ?? 0).toFixed(2)} - KSh {Number(predictionData?.day_high ?? 0).toFixed(2)}</span></div>
            <div className="info-row"><span>52 Week Range</span><span>KSh {Number(predictionData?.['12m_low'] ?? predictionData?.low12m ?? 0).toFixed(2)} - KSh {Number(predictionData?.['12m_high'] ?? predictionData?.high12m ?? 0).toFixed(2)}</span></div>
            <div className="info-row"><span>Volume</span><span>{formatVolume(predictionData?.volume)}</span></div>
          </div>
        </div>
      )}

      {/* Trends Card */}
      {trendsData && (
        <div className="card trends-card">
          <div className="trends-header"><span>30-Day Trend Analysis</span></div>
          <div className="trend-summary" style={{ borderColor: trendsData.trend === 'upward' ? '#4caf50' : trendsData.trend === 'downward' ? '#f44336' : '#ff9800' }}>
            <div className="trend-emoji">{trendsData.trend === 'upward' ? '📈' : trendsData.trend === 'downward' ? '📉' : '➡️'}</div>
            <div className="trend-title">{trendsData.trend?.toUpperCase() || 'N/A'} TREND</div>
          </div>

          <div className="info-rows">
            <div className="info-row"><span>Period</span><span>{trendsData.period_days} days</span></div>
            <div className="info-row"><span>Highest</span><span>KSh {Number(trendsData.highest_price).toFixed(2)}</span></div>
            <div className="info-row"><span>Lowest</span><span>KSh {Number(trendsData.lowest_price).toFixed(2)}</span></div>
            <div className="info-row"><span>Average</span><span>KSh {Number(trendsData.average_price).toFixed(2)}</span></div>
            <div className="info-row"><span>Price Change</span><span>{trendsData.price_change >= 0 ? '+' : ''}KSh {Number(trendsData.price_change).toFixed(2)}</span></div>
          </div>
        </div>
      )}

      {/* Technical Indicators */}
      {predictionData?.features_used && (
        <div className="card tech-card">
          <div className="card-title">Technical Indicators</div>
          <div className="info-rows">
            <div className="info-row"><span>5-Day MA</span><span>KSh {Number(predictionData.features_used.ma_5).toFixed(2)}</span></div>
            <div className="info-row"><span>10-Day MA</span><span>KSh {Number(predictionData.features_used.ma_10).toFixed(2)}</span></div>
            <div className="info-row"><span>Daily Volatility</span><span>{Number(predictionData.features_used.daily_volatility).toFixed(2)}</span></div>
            <div className="info-row"><span>From 52W Low</span><span style={{ color: predictionData.features_used.pct_from_12m_low >=0 ? '#4caf50' : '#f44336' }}>{predictionData.features_used.pct_from_12m_low >=0 ? '+' : ''}{Number(predictionData.features_used.pct_from_12m_low).toFixed(2)}%</span></div>
            <div className="info-row"><span>From 52W High</span><span style={{ color: predictionData.features_used.pct_from_12m_high >=0 ? '#4caf50' : '#f44336' }}>{predictionData.features_used.pct_from_12m_high >=0 ? '+' : ''}{Number(predictionData.features_used.pct_from_12m_high).toFixed(2)}%</span></div>
          </div>
        </div>
      )}
    </div>
  );
}
