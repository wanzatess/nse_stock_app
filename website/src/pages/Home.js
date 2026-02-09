import { useEffect, useState } from "react";
import { getMarketOverview } from "../api/apiService";

export default function Home() {
  const [data, setData] = useState(null);

  useEffect(() => {
    getMarketOverview().then(res => setData(res.data));
  }, []);

  if (!data) return <div>Loading...</div>;

  return (
    <div>
      <h1>NSE Market Overview</h1>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}
