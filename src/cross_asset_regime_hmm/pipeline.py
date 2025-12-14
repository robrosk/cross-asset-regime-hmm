from .data_source import load_ohlcv_yfinance
from .features import add_log_return, add_rolling_vol


def run_pipeline(symbols: list[str], start: str, end: str, vol_window: int = 10, n_states: int = 3):
    """
    Download data for each symbol, build features, and return per-symbol matrices.
    Raises on missing data/columns so issues are surfaced immediately.
    """
    data = load_ohlcv_yfinance(symbols, start, end)
    if not data:
        raise ValueError("No data returned from load_ohlcv_yfinance.")

    results = {}
    required_cols = ["log_return", "roll_vol"]

    for sym, df in data.items():
        if df is None or df.empty:
            raise ValueError(f"{sym}: no data returned.")

        df = df.copy()
        if "price" not in df.columns:
            raise KeyError(f"{sym}: missing 'price' column after standardization. Columns: {list(df.columns)}")

        df = add_log_return(df)                          # adds df["log_return"]
        df = add_rolling_vol(df, window=vol_window)      # adds df["roll_vol"]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"{sym}: missing columns after feature engineering: {missing}. Columns: {list(df.columns)}")

        df_feat = df.dropna(subset=required_cols).copy()
        if df_feat.empty:
            raise ValueError(f"{sym}: all rows dropped during feature cleaning.")

        X = df_feat[required_cols].to_numpy()

        results[sym] = {
            "symbol": sym,
            "features": X,
            "data": df_feat,
        }

    return results