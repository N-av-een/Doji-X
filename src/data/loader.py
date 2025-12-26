import yfinance as yf
from pathlib import Path

def download_5m_60d(symbol: str):
    df = yf.download(
        symbol,
        interval="5m",
        period="60d",
        auto_adjust=False,
        progress=False,
    )
    return df

def load_gold_5m_60d():
    # Try spot first (may fail), then fallback to futures
    candidates = ["XAUUSD=X", "GC=F"]

    last_err = None
    for sym in candidates:
        try:
            df = download_5m_60d(sym)
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume"
                })
                df = df[["open", "high", "low", "close", "volume"]].dropna()
                df.index.name = "datetime"
                return sym, df
        except Exception as e:
            last_err = e

    raise ValueError(f"No data downloaded from any candidate ticker. Last error: {last_err}")

if __name__ == "__main__":
    symbol_used, df = load_gold_5m_60d()

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{symbol_used.replace('=','_').replace('^','')}_5m_60d.csv"
    df.to_csv(out_path)

    print(f"✅ Downloaded {len(df)} rows using ticker: {symbol_used}")
    print(f"✅ Saved to: {out_path}")
