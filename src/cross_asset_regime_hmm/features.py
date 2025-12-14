from functools import reduce
import numpy as np
import pandas as pd

def align_to_common_dates(data: dict, feature_cols=("log_return", "roll_vol")) -> dict:
    """
    Filter each symbol's data to only include dates present in ALL symbols (intersection).
    Rebuilds features from aligned rows to guarantee consistency.
    """
    indices = [d["data"].index for d in data.values()]
    common_idx = reduce(lambda a, b: a.intersection(b), indices)

    common_idx = common_idx.sort_values()
    if len(common_idx) == 0:
        raise ValueError("No common dates across all symbols after alignment.")

    aligned = {}
    for sym, d in data.items():
        df_aligned = d["data"].loc[common_idx].copy()

        missing = [c for c in feature_cols if c not in df_aligned.columns]
        if missing:
            raise KeyError(f"{sym}: missing feature columns after alignment: {missing}")

        aligned[sym] = {
            "symbol": sym,
            "features": df_aligned[list(feature_cols)].to_numpy(),  # shape (n, d)
            "data": df_aligned,
        }

    return aligned

def add_log_return(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    out = df.copy()
    out["log_return"] = np.log(out[price_col]).diff()
    return out

def add_rolling_vol(df: pd.DataFrame, ret_col: str = "log_return", window: int = 10) -> pd.DataFrame:
    out = df.copy()
    out["roll_vol"] = out[ret_col].rolling(window).std()
    return out

def build_feature_matrix(df: pd.DataFrame, cols=("log_return", "roll_vol")):
    return df.loc[:, cols].dropna().to_numpy()

def add_features(data: pd.DataFrame, required_cols=("log_return", "roll_vol"), vol_window: int = 10, n_states: int = 3):
    results = {}

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
