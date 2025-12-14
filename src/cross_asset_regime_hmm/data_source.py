import pandas as pd
import yfinance as yf


def load_ohlcv_yfinance(symbols: list[str], start: str, end: str):
    """
    Works for both stocks (e.g., 'SPY') and crypto pairs (e.g., 'BTC-USD').
    Returns standardized OHLCV columns.
    """
    data = yf.download(symbols, start=start, end=end)
    
    return data

def load_ohlcv_crypto(symbol: str, start: str, end: str): 
    pass


if __name__ == "__main__":
    data = load_ohlcv_yfinance(['SPY', 'QQQ', 'USD-BTC'], '2025-01-01', '2025-12-13')

    # available syntax for getting the data
    print(data['Close']['SPY'])           
    print(data[('Close', 'SPY')]) 
    print(data['Close']['USD-BTC'])


