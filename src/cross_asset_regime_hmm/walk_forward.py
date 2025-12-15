from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data_source import load_ohlcv_yfinance
from .features import add_features, align_to_common_dates
from .hmm_model import RegimeHMM
from .interpret import label_states_cross_asset


def run_pipeline_walk_forward(
    symbols: List[str],
    start: str,
    end: str,
    vol_window: int = 10,
    n_states: int = 3,
    covariance_type: str = "full",
    window_size: int = 252,
) -> Dict:
    """
    Walk-forward pipeline to prevent look-ahead bias.

    For each time step t >= window_size:
      - fit scaler + HMM on the past `window_size` rows (historical only)
      - predict the state for day t using that model (one-step ahead)

    Returns a dict similar to `run_pipeline`, with wide["state"]/wide["regime"]
    populated from `window_size` onward (earlier rows remain NaN).
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    # --- Load & feature engineering (same as classic pipeline) ---
    data = load_ohlcv_yfinance(symbols, start, end)
    if not data:
        raise ValueError("No data returned from load_ohlcv_yfinance.")

    feature_cols_per_asset = ["log_return", "roll_vol"]
    data = add_features(data, feature_cols_per_asset, vol_window, n_states)
    data = align_to_common_dates(data)

    common_index = data[symbols[0]]["data"].index
    wide = pd.DataFrame(index=common_index)

    wide_feature_cols: List[str] = []
    for sym in symbols:
        df = data[sym]["data"]
        wide[f"{sym}_log_return"] = df["log_return"].to_numpy()
        wide[f"{sym}_roll_vol"] = df["roll_vol"].to_numpy()
        wide_feature_cols += [f"{sym}_log_return", f"{sym}_roll_vol"]

    X = wide[wide_feature_cols].to_numpy()
    n_rows = X.shape[0]
    if n_rows <= window_size:
        raise ValueError(
            f"Not enough rows ({n_rows}) for walk-forward window_size={window_size}."
        )

    states: list[int] = [np.nan] * window_size  # first window is undefined
    regimes: list[str | float] = [np.nan] * window_size

    last_model: RegimeHMM | None = None
    last_scaler: StandardScaler | None = None
    last_label_map: dict[int, str] | None = None

    # Precompute feature index layout for labeling
    n_assets = len(symbols)
    return_indices = [2 * i for i in range(n_assets)]
    vol_indices = [2 * i + 1 for i in range(n_assets)]

    for t in range(window_size, n_rows):
        X_hist = X[t - window_size : t]

        scaler = StandardScaler().fit(X_hist)
        X_hist_scaled = scaler.transform(X_hist)

        model = RegimeHMM(n_states=n_states, covariance_type=covariance_type)
        model.fit(X_hist_scaled)

        X_next_scaled = scaler.transform(X[t].reshape(1, -1))
        state_t = int(model.predict_states(X_next_scaled)[0])

        label_map = label_states_cross_asset(
            means=model.means_,
            n_states=n_states,
            return_indices=return_indices,
            vol_indices=vol_indices,
        )
        regime_t = label_map[state_t]

        states.append(state_t)
        regimes.append(regime_t)

        last_model = model
        last_scaler = scaler
        last_label_map = label_map

    wide["state"] = states
    wide["regime"] = regimes

    return {
        "wide": wide,
        "per_asset": data,
        "model": last_model,
        "scaler": last_scaler,
        "label_map": last_label_map or {},
        "wide_feature_cols": wide_feature_cols,
        "mode": "walk_forward",
        "window_size": window_size,
    }

