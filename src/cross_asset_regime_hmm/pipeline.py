from .data_source import load_ohlcv_yfinance
from .features import add_features, align_to_common_dates
from .hmm_model import RegimeHMM
from .interpret import label_states_cross_asset

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def run_pipeline(
    symbols: list[str],
    start: str,
    end: str,
    vol_window: int = 10,
    n_states: int = 3,
):
    """
    Download data for each symbol, build features, align to common dates (intersection),
    train a cross-asset HMM, label regimes, and return results.
    """
    data = load_ohlcv_yfinance(symbols, start, end)
    if not data:
        raise ValueError("No data returned from load_ohlcv_yfinance.")

    feature_cols_per_asset = ["log_return", "roll_vol"]
    data = add_features(data, feature_cols_per_asset, vol_window, n_states)

    # Align all symbols to common dates (intersection) and rebuild per-asset feature matrices
    data = align_to_common_dates(data)

    # Build a wide dataframe (guarantees date/row alignment)
    common_index = data[symbols[0]]["data"].index
    wide = pd.DataFrame(index=common_index)

    # Keep a consistent column order: (ret, vol) per symbol
    wide_feature_cols: list[str] = []
    for sym in symbols:
        df = data[sym]["data"]
        wide[f"{sym}_log_return"] = df["log_return"].to_numpy()
        wide[f"{sym}_roll_vol"] = df["roll_vol"].to_numpy()
        wide_feature_cols += [f"{sym}_log_return", f"{sym}_roll_vol"]

    # Feature matrix for HMM: shape (n_timesteps, 2 * n_assets)
    X = wide[wide_feature_cols].to_numpy()

    # Scale features for stable training
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train HMM + decode states
    hmm_model = RegimeHMM(n_states=n_states, covariance_type="full")
    hmm_model.fit(X_scaled)

    states = hmm_model.predict_states(X_scaled)
    wide["state"] = states

    # Cross-asset labeling using model means and known feature layout
    n_assets = len(symbols)
    return_indices = [2 * i for i in range(n_assets)]      # indices of each asset's return feature in X
    vol_indices = [2 * i + 1 for i in range(n_assets)]     # indices of each asset's vol feature in X

    label_map = label_states_cross_asset(
        means=hmm_model.means_,          # in scaled feature space (fine for ranking/labeling)
        n_states=n_states,
        return_indices=return_indices,
        vol_indices=vol_indices,
    )
    wide["regime"] = wide["state"].map(label_map)

    return {
        "wide": wide,
        "per_asset": data,
        "model": hmm_model,
        "scaler": scaler,
        "label_map": label_map,
        "wide_feature_cols": wide_feature_cols,
    }
