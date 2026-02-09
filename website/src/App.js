import React, { useState } from "react";
import MarketOverview from "./pages/MarketOverview";
import SearchScreen from "./pages/Search";
import TopStocks from "./pages/TopStocks";
import StockDetail from "./pages/StockDetail";
import "./styles/main.css";
import "./styles/MarketOverview.css"; // Ensure global dark theme is applied

function App() {
  // 'home', 'search', 'top-stocks', 'market'
  const [activeTab, setActiveTab] = useState("home");
  const [selectedStock, setSelectedStock] = useState(null);

  // Placeholder for when a user clicks a stock in Search
  const handleStockClick = (stock) => {
    // Open stock detail view
    setSelectedStock(stock);
    setActiveTab("detail");
  };

  return (
    <div className="app-container">
      {/* --- Screen Content Switcher --- */}
      <div className="content-area">
        {activeTab === "home" && <MarketOverview />}
        {activeTab === "market" && <MarketOverview />}
        {activeTab === "detail" && selectedStock && (
          <StockDetail symbol={selectedStock.symbol} name={selectedStock.name} onBack={() => setActiveTab("search")} />
        )}
        
        {activeTab === "search" && (
          <SearchScreen onStockClick={handleStockClick} />
        )}

        {/* Top Stocks Screen */}
        {activeTab === "top-stocks" && (
          <TopStocks onStockClick={handleStockClick} />
        )}
      </div>

      {/* --- Bottom Navigation Bar --- */}
      <nav className="bottom-nav">
        <div 
          className={`nav-item ${activeTab === "home" ? "active" : ""}`}
          onClick={() => setActiveTab("home")}
        >
          <span className="nav-icon">🏠</span>
          <span className="nav-label">Home</span>
        </div>

        <div 
          className={`nav-item ${activeTab === "search" ? "active" : ""}`}
          onClick={() => setActiveTab("search")}
        >
          <span className="nav-icon">🔍</span>
          <span className="nav-label">Search</span>
        </div>

        <div 
          className={`nav-item ${activeTab === "top-stocks" ? "active" : ""}`}
          onClick={() => setActiveTab("top-stocks")}
        >
          <span className="nav-icon">📊</span>
          <span className="nav-label">Top Stocks</span>
        </div>

        <div 
          className={`nav-item ${activeTab === "market" ? "active" : ""}`} 
          onClick={() => setActiveTab("market")}
        >
          <span className="nav-icon">📅</span>
          <span className="nav-label">Market</span>
        </div>
      </nav>
    </div>
  );
}

export default App;