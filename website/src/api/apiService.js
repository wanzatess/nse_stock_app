import axios from "axios";

const baseURL = (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'))
  ? 'http://127.0.0.1:8001'
  : 'https://nse-stock-app.onrender.com';

const api = axios.create({
  baseURL,
  timeout: 10000,
});

export const getMarketOverview = () => api.get("/market-overview");
export const getTopStocks = (criteria="gainers") =>
  api.get(`/top-stocks?criteria=${criteria}`);
export const predictStock = (symbol) =>
  api.post("/predict", { symbol });
export const getTrends = (symbol, days = 30) => api.get(`/trends/${symbol}?days=${days}`);
export const getHistory = (symbol, days = 30) => api.get(`/history/${symbol}?days=${days}`);
