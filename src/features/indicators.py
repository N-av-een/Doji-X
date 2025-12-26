import pandas as pd
import numpy as np

def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Heikin Ashi OHLC columns: ha_open, ha_high, ha_low, ha_close
    Expects columns: open, high, low, close
    """
    ha = df.copy()

    ha_close = (ha["open"] + ha["high"] + ha["low"] + ha["close"]) / 4.0

    ha_open = np.zeros(len(ha))
    ha_open[0] = (ha["open"].iloc[0] + ha["close"].iloc[0]) / 2.0

    for i in range(1, len(ha)):
        ha_open[i] = (ha_open[i - 1] + ha_close.iloc[i - 1]) / 2.0

    ha_open = pd.Series(ha_open, index=ha.index)
    ha_high = pd.concat([ha["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low  = pd.concat([ha["low"],  ha_open, ha_close], axis=1).min(axis=1)

    ha["ha_open"] = ha_open
    ha["ha_high"] = ha_high
    ha["ha_low"] = ha_low
    ha["ha_close"] = ha_close

    return ha

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1/period, adjust=False).mean()

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def is_doji(ha_df: pd.DataFrame, body_to_range: float = 0.1) -> pd.Series:
    """
    Doji rule on Heikin Ashi candles:
    body <= body_to_range * range
    """
    body = (ha_df["ha_close"] - ha_df["ha_open"]).abs()
    rng = (ha_df["ha_high"] - ha_df["ha_low"]).replace(0, np.nan)
    return body <= (body_to_range * rng)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds HA + EMA50/200 + RSI14 + ATR14 + MACD + doji flag
    """
    out = heikin_ashi(df)

    out["ema_50"] = ema(out["close"], 50)
    out["ema_200"] = ema(out["close"], 200)
    out["rsi_14"] = rsi(out["close"], 14)
    out["atr_14"] = atr(out, 14)

    m_line, s_line, h = macd(out["close"])
    out["macd"] = m_line
    out["macd_signal"] = s_line
    out["macd_hist"] = h

    out["ha_green"] = out["ha_close"] > out["ha_open"]
    out["ha_red"] = out["ha_close"] < out["ha_open"]
    out["ha_doji"] = is_doji(out)

    return out