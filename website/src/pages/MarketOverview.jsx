import React, { useEffect, useState } from "react";
import {
  getMarketOverview,
  getTopStocks,
} from "../api/apiService";
import "../styles/MarketOverview.css";

const toNumber = (val) => {
  if (val === null || val === undefined) return 0;
  if (typeof val === "string") {
    return parseFloat(val.replace("%", "")) || 0;
  }
  return Number(val) || 0;
};

const MarketOverview = () => {
  const [market, setMarket] = useState(null);
  const [topGainers, setTopGainers] = useState([]);
  const [buySignals, setBuySignals] = useState([]);
  const [loading, setLoading] = useState(true);

  const loadMarketData = async () => {
    setLoading(true);
    try {
      const [marketRes, gainersRes, buysRes] = await Promise.all([
        getMarketOverview(),
        getTopStocks("gainers"),
        getTopStocks("buy_signals"),
      ]);

      setMarket(marketRes.data);
      setTopGainers(gainersRes.data.stocks || gainersRes.data || []);
      setBuySignals(buysRes.data.stocks || buysRes.data || []);
    } catch (err) {
      console.error("❌ Market overview fetch error:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadMarketData();
  }, []);

  if (loading) {
    return <div className="market-loading">Loading...</div>;
  }

  const avgChange = toNumber(market?.average_change);
  const isPositive = avgChange >= 0;

  return (
    <div className="market-overview">
      {/* ===== Single Big Market Overview Card ===== */}
      <div className="card overview-card">
        {/* Header Row: Title + Percentage Badge + Refresh */}
        <div className="card-header">
          <h2>Market Overview</h2>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <button
              onClick={loadMarketData}
              title="Refresh"
              style={{
                background: "transparent",
                border: "none",
                color: "#fff",
                cursor: "pointer",
                fontSize: 18,
              }}
            >
              🔄
            </button>
            <div className={`badge ${isPositive ? "bg-green" : "bg-red"}`}>
              {isPositive ? "+" : ""}{avgChange.toFixed(2)}%
            </div>
          </div>
        </div>

        {/* Stats Row: Gainers | Losers | Unchanged */}
        <div className="stats-row">
          <div className="stat-item">
            <span className="stat-value text-green">{market?.gainers || 0}</span>
            <span className="stat-label">Gainers</span>
          </div>
          
          {/* Vertical Divider */}
          <div className="stat-divider"></div>

          <div className="stat-item">
            <span className="stat-value text-red">{market?.losers || 0}</span>
            <span className="stat-label">Losers</span>
          </div>

          {/* Vertical Divider */}
          <div className="stat-divider"></div>

          <div className="stat-item">
            <span className="stat-value text-orange">{market?.unchanged || 0}</span>
            <span className="stat-label">Unchanged</span>
          </div>
        </div>
      </div>

      {/* ===== Top Gainers List (Rounded Widgets) ===== */}
      <div className="section-container">
        <h3>Top Gainers</h3>
        
        <div className="stock-list">
          {topGainers.map((stock) => {
             const change = toNumber(stock.change_percent);
             const isPos = change >= 0;
             
             return (
              <div key={stock.symbol} className="stock-card">
                <div className="stock-info">
                  <span className="stock-symbol">{stock.symbol}</span>
                  <span className="stock-name">{stock.name || "Company Name"}</span>
                </div>

                <div className="stock-stats">
                  <span className="stock-price">{toNumber(stock.current_price).toFixed(2)}</span>
                  <div className={`change-pill ${isPos ? "pill-green" : "pill-red"}`}>
                    {isPos ? "+" : ""}{change.toFixed(2)}%
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
      
      {/* ===== Buy Signals ===== */}
      <div className="section-container" style={{ marginTop: 24 }}>
        <h3>Buy Signals</h3>

        <div className="stock-list">
          {buySignals.length === 0 ? (
            <div className="card" style={{ padding: 16 }}>
              <div style={{ color: '#888' }}>No buy signals available</div>
            </div>
          ) : (
            buySignals.map((stock) => {
              const change = toNumber(stock.change_percent);
              const isPos = change >= 0;
              return (
                <div key={stock.symbol} className="stock-card">
                  <div className="stock-info">
                    <span className="stock-symbol">{stock.symbol}</span>
                    <span className="stock-name">{stock.name || "Company Name"}</span>
                  </div>

                  <div className="stock-stats">
                    <span className="stock-price">{toNumber(stock.current_price).toFixed(2)}</span>
                    <div className={`change-pill ${isPos ? "pill-green" : "pill-red"}`}>
                      {isPos ? "+" : ""}{change.toFixed(2)}%
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};

export default MarketOverview;