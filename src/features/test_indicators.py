import pandas as pd

def read_ohlc_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Common datetime column names
    for c in ["datetime", "Datetime", "Date", "Timestamp", "time", "Time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
            df = df.set_index(c).sort_index()
            df.index.name = "datetime"
            break
    else:
        # If none found, assume the FIRST column is datetime/index-like
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
        if df[first_col].isna().all():
            raise ValueError(f"First column '{first_col}' is not parseable as datetime. Columns: {list(df.columns)}")

        df = df.set_index(first_col).sort_index()
        df.index.name = "datetime"

    # Normalize column names
    df = df.rename(columns={c: c.lower() for c in df.columns})

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Found columns: {list(df.columns)}")

    if "volume" not in df.columns:
        df["volume"] = 0

    df = df[["open", "high", "low", "close", "volume"]].dropna()
    return df
