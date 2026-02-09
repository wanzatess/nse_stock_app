import React, { useEffect, useState } from "react";
import axios from "axios";
import "../styles/TopStocks.css";

const API_BASE = "https://nse-stock-app.onrender.com";

const TABS = [
  { key: "gainers", label: "Gainers", icon: "▲", color: "#4caf50" },
  { key: "losers", label: "Losers", icon: "▼", color: "#f44336" },
  { key: "volume", label: "Volume", icon: "▮", color: "#2196f3" },
  { key: "buy_signals", label: "Buy Signals", icon: "★", color: "#ff9800" },
];

export default function TopStocks({ onStockClick }) {
  const [activeTab, setActiveTab] = useState("gainers");
  const [data, setData] = useState({ gainers: [], losers: [], volume: [], buy_signals: [] });
  const [loading, setLoading] = useState({ gainers: true, losers: true, volume: true, buy_signals: true });
  const [error, setError] = useState(null);

  useEffect(() => {
    // load all tabs initially (like Flutter implementation)
    TABS.forEach(t => load(t.key));
  }, []);

  async function load(criteria) {
    setLoading(prev => ({ ...prev, [criteria]: true }));
    setError(null);
    try {
      const res = await axios.get(`${API_BASE}/top-stocks?criteria=${criteria}&limit=20`, { timeout: 15000 });
      const list = Array.isArray(res.data) ? res.data : res.data.stocks || [];
      setData(prev => ({ ...prev, [criteria]: list }));
    } catch (e) {
      console.error(e);
      setError("Error loading data");
    } finally {
      setLoading(prev => ({ ...prev, [criteria]: false }));
    }
  }

  const currentList = data[activeTab] || [];

  return (
    <div className="market-overview">
      <h2>Top Stocks</h2>

      <div style={{ display: "flex", gap: 8, marginBottom: 12, overflowX: "auto" }}>
        {TABS.map(t => (
          <button
            key={t.key}
            onClick={() => setActiveTab(t.key)}
            style={{
              padding: "8px 12px",
              borderRadius: 12,
              border: activeTab === t.key ? `2px solid ${t.color}` : "1px solid #333",
              background: activeTab === t.key ? "#1E1E1E" : "transparent",
              color: "#fff",
            }}
          >
            {t.icon} {t.label}
          </button>
        ))}
      </div>

      {error && <div style={{ color: "#f44336" }}>{error}</div>}

      {(loading[activeTab] || loading.gainers || loading.losers || loading.volume || loading.buy_signals) && (
        <div className="market-loading">Loading...</div>
      )}

      {!loading[activeTab] && currentList.length === 0 && (
        <div style={{ textAlign: "center", padding: 40, color: "#888" }}>
          <div style={{ fontSize: 48 }}>📥</div>
          <div style={{ fontSize: 18 }}>No data available</div>
        </div>
      )}

      <div className="stock-list">
        {currentList.map((s, idx) => {
          const rank = idx + 1;
          const change = s.change ?? 0;
          const changePercent = s.change_percent ?? 0;
          const isPositive = change >= 0;
          return (
            <div
              key={s.symbol || idx}
              className="stock-card"
              onClick={() => onStockClick && onStockClick({ symbol: s.symbol, name: s.name })}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <div style={{
                  width: 40,
                  height: 40,
                  borderRadius: 8,
                  background: (activeTab === 'gainers' ? '#4caf50' : activeTab === 'losers' ? '#f44336' : activeTab === 'volume' ? '#2196f3' : '#ff9800') + '22',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}>
                  <strong style={{ color: activeTab === 'gainers' ? '#4caf50' : activeTab === 'losers' ? '#f44336' : activeTab === 'volume' ? '#2196f3' : '#ff9800' }}>#{rank}</strong>
                </div>

                <div className="stock-info">
                  <div className="stock-symbol">{s.symbol}</div>
                  <div className="stock-name">{s.name}</div>
                  {activeTab === 'volume' && (
                    <div style={{ fontSize: 12, color: '#b0b0b0', marginTop: 6 }}>Vol: {formatVolume(s.volume)}</div>
                  )}
                </div>
              </div>

              <div className="stock-stats">
                <div className="stock-price">KSh {Number(s.current_price).toFixed(2)}</div>
                <div className={`change-pill ${isPositive ? 'pill-green' : 'pill-red'}`}>
                  <span style={{ marginRight: 8 }}>{isPositive ? '▲' : '▼'}</span>
                  {`${isPositive ? '+' : ''}${Number(changePercent).toFixed(2)}%`}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function formatVolume(volume) {
  const v = Number(volume) || 0;
  if (v >= 10000000) return `${(v / 10000000).toFixed(2)}Cr`;
  if (v >= 100000) return `${(v / 100000).toFixed(2)}L`;
  if (v >= 1000) return `${(v / 1000).toFixed(2)}K`;
  return String(v);
}
