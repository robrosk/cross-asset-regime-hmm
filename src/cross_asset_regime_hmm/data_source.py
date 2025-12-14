from typing import Optional

import pandas as pd
import yfinance as yf


def _flatten_columns(df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    """Flatten possible MultiIndex columns returned by yfinance."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    # If only one ticker level, drop it to keep standard OHLCV names.
    if df.columns.nlevels == 2 and df.columns.get_level_values(-1).nunique() == 1:
        return df.droplevel(-1, axis=1)

    # Fallback: join levels with underscore.
    df = df.copy()
    df.columns = ["_".join([str(c) for c in tup if c not in ("", None)]) for tup in df.columns]
    return df


def standardize_ohlcv(df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    df = _flatten_columns(df, symbol)

    if "Adj Close" in df.columns:
        df["price"] = df["Adj Close"]
    elif "Close" in df.columns:
        df["price"] = df["Close"]
    else:
        raise KeyError("Could not find Close/Adj Close columns after flattening")

    return df


def load_ohlcv_yfinance(symbols: list[str], start: str, end: str) -> dict[str, "pd.DataFrame"]:
    """
    Works for both stocks (e.g., 'SPY') and crypto pairs (e.g., 'BTC-USD').
    Returns standardized OHLCV columns in a dictionary.
    """
    out = {}
    for sym in symbols:
        df = yf.download(sym, start=start, end=end, progress=False, group_by="column")
        df = df.dropna(how="all")
        out[sym] = standardize_ohlcv(df, sym)

    return out


if __name__ == "__main__":
    data = load_ohlcv_yfinance(["SPY", "QQQ", "BTC-USD"], "2025-01-01", "2025-12-13")
    print(data["BTC-USD"][["price"]].head())