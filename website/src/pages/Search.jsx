import React, { useEffect, useState } from "react";
import axios from "axios";
import "../styles/search.css";

const API_BASE = "https://nse-stock-app.onrender.com";

const SearchScreen = ({ onStockClick }) => {
  const [stocks, setStocks] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStocks();
  }, []);

  const fetchStocks = async () => {
    try {
      const res = await axios.get(`${API_BASE}/stocks`);
      let stockList = [];
      if (Array.isArray(res.data)) {
        stockList = res.data;
      } else if (res.data && Array.isArray(res.data.stocks)) {
        stockList = res.data.stocks;
      }
      setStocks(stockList);
    } catch (error) {
      console.error("Error fetching stocks:", error);
      setStocks([]);
    } finally {
      setLoading(false);
    }
  };

  const safeStocks = Array.isArray(stocks) ? stocks : [];

  const filteredStocks = safeStocks.filter((stock) => {
    const term = searchTerm.toLowerCase();
    const symbol = (stock.code || stock.symbol || "").toLowerCase();
    const name = (stock.name || "").toLowerCase();
    return symbol.includes(term) || name.includes(term);
  });

  return (
    <div className="search-screen">
      <div className="search-header">
        <h2>Search Stocks</h2>

        <div className="search-bar-container">
          <span className="search-icon">🔍</span>
          <input
            type="text"
            placeholder="Search by symbol or company name"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="search-input"
          />
          {searchTerm && (
            <button
              className="clear-btn"
              onClick={() => setSearchTerm("")}
              aria-label="Clear search"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      <div className="search-list">
        {loading ? (
          <div className="search-loading">
            <div className="spinner-small" />
            <div>Loading...</div>
          </div>
        ) : filteredStocks.length > 0 ? (
          filteredStocks.map((stock, index) => {
            const symbol = stock.code || stock.symbol || "?";
            const firstLetter = (symbol[0] || "?").toUpperCase();
            return (
              <div
                key={index}
                className="search-item"
                onClick={() => onStockClick({ symbol, name: stock.name })}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <div className="stock-avatar">{firstLetter}</div>
                  <div className="search-item-left">
                    <div className="item-symbol">{symbol}</div>
                    <div className="item-name">{stock.name || "Unknown Name"}</div>
                  </div>
                </div>

                <div className="stock-action">
                  <span className="arrow-icon">›</span>
                </div>
              </div>
            );
          })
        ) : (
          <div className="no-results">
            <div style={{ fontSize: 56, color: "#666", marginBottom: 12 }}>🔍</div>
            <div style={{ color: "#888" }}>{searchTerm ? `No results for "${searchTerm}"` : "No stocks found"}</div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SearchScreen;