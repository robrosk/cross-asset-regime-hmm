import numpy as np
import pandas as pd

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
