from .data_source import load_ohlcv_yfinance
from .features import add_log_return, add_rolling_vol


def run_pipeline(symbols: list[str], start: str, end: str, vol_window: int = 10, n_states: int = 3):
    # load df
    df = load_ohlcv_yfinance()
    df = add_log_return(df)                  # df["log_return"]
    df = add_rolling_vol(df, window=10)      # df["roll_vol"]
    df = add_rolling_vol(df, window=10)      # df["roll_vol"]

    df_feat = df.dropna(subset=["log_return", "roll_vol"]).copy()
    X = df_feat[["log_return", "roll_vol"]].to_numpy()

    print(X.shape)
    print(df_feat[["price","log_return","roll_vol"]].tail())
    # add features
    # build X
    # fit hmm
    # decode
    # label states
    # return result df + model + label map