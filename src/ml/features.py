from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/GC_F_5m_60d_with_indicators.csv")
OUTPUT_PATH = Path("data/processed/ml_dataset.csv")

# Predict direction after N bars (3 bars = 15 minutes on 5m candles)
HORIZON_BARS = 3

def main():
    df = pd.read_csv(INPUT_PATH)

    # Ensure datetime column
    if "datetime" not in df.columns:
        df = df.rename(columns={df.columns[0]: "datetime"})

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").set_index("datetime")

    # Create target: future return and direction label
    df["future_close"] = df["close"].shift(-HORIZON_BARS)
    df["future_ret"] = (df["future_close"] - df["close"]) / df["close"]
    df["y"] = (df["future_ret"] > 0).astype(int)

    # Feature selection (safe: use only columns that exist)
    candidate_features = [
        "open","high","low","close","volume",
        "ha_open","ha_high","ha_low","ha_close",
        "ema_50","ema_200","rsi_14","atr_14",
        "macd","macd_signal","macd_hist",
        "ha_green","ha_red","ha_doji"
    ]
    features = [c for c in candidate_features if c in df.columns]

    out = df[features + ["y", "future_ret"]].dropna().copy()

    # Remove last horizon rows (no label)
    if len(out) > HORIZON_BARS:
        out = out.iloc[:-HORIZON_BARS]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH)

    print("ML dataset saved:", OUTPUT_PATH)
    print("Rows:", len(out))
    print("Features:", len(features))
    print("Feature sample:", features[:10])

if __name__ == "__main__":
    main()
