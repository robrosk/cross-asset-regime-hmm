import pandas as pd
import yfinance as yf

def standardize_ohlcv(df):
    df = df.copy()
    if "Adj Close" in df.columns:
        df["price"] = df["Adj Close"]
    else:
        df["price"] = df["Close"]
    return df

def load_ohlcv_yfinance(symbols: list[str], start: str, end: str) -> dict[str, "pd.DataFrame"]:
    """
    Works for both stocks (e.g., 'SPY') and crypto pairs (e.g., 'BTC-USD').
    Returns standardized OHLCV columns in a dictionary.
    """
    out = {}
    for sym in symbols:
        df = yf.download(sym, start=start, end=end, progress=False)
        out[sym] = standardize_ohlcv(df.dropna(how="all"))
        
    return out


if __name__ == "__main__":
    data = load_ohlcv_yfinance(['SPY', 'QQQ', 'BTC-USD'], '2025-01-01', '2025-12-13')
    
    print(data['BTC-USD']['price'])


