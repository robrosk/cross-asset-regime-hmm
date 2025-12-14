"""
Prediction utilities for Cross-Asset Regime HMM.

API:
  - predict_today(res) -> dict
  - predict_tomorrow(res) -> dict

Both return compact dictionaries intended for a CLI / API layer:
  - regime posterior today
  - regime posterior tomorrow (p_today @ A)
  - uncertainty score (max(p_today))
  - expected next-day return per asset (optional; mixture of regime means)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def _get_X_scaled(res: dict) -> np.ndarray:
    wide: pd.DataFrame = res["wide"]
    feature_cols: list[str] = res["wide_feature_cols"]
    scaler = res["scaler"]
    X = wide[feature_cols].to_numpy()
    return scaler.transform(X)


def _state_labels(res: dict, n_states: int) -> list[str]:
    label_map: dict[int, str] = res.get("label_map", {})
    return [label_map.get(i, f"State{i}") for i in range(n_states)]


def _posterior_today(res: dict) -> np.ndarray:
    model = res["model"]
    X_scaled = _get_X_scaled(res)
    post = model.predict_probs(X_scaled)
    if post.shape[0] == 0:
        raise ValueError("No rows available to compute posterior.")
    return post[-1]


def _expected_next_day_returns(res: dict, p_next: np.ndarray) -> Optional[dict[str, float]]:
    """
    Expected next-day log return per asset using mixture of emission means.

    Assumes features are ordered [sym1_ret, sym1_vol, sym2_ret, sym2_vol, ...]
    and the HMM was fit on scaled features.
    """
    wide_feature_cols: list[str] = res.get("wide_feature_cols", [])
    if not wide_feature_cols:
        return None

    model = res["model"]
    scaler = res["scaler"]

    return_indices = list(range(0, len(wide_feature_cols), 2))
    mu_scaled = model.means_[:, return_indices]         # (n_states, n_assets) in scaled space
    E_ret_scaled = p_next @ mu_scaled                   # (n_assets,)

    means = scaler.mean_[return_indices]
    scales = scaler.scale_[return_indices]
    E_ret = E_ret_scaled * scales + means               # back to original units

    sym_names = [col.replace("_log_return", "") for col in wide_feature_cols[::2]]
    return {sym: float(r) for sym, r in zip(sym_names, E_ret)}


def predict_today(res: dict) -> dict[str, Any]:
    """
    Predict regime distribution for the last observed day in `res`.
    """
    wide: pd.DataFrame = res["wide"]
    if wide.empty:
        raise ValueError("wide is empty; cannot predict.")

    p_today = _posterior_today(res)
    n_states = int(p_today.shape[0])
    labels = _state_labels(res, n_states)

    today_date = wide.index[-1]
    today_state = int(np.argmax(p_today))
    confidence = float(np.max(p_today))
    uncertainty = float(1.0 - confidence)
    # entropy in nats; normalized to [0,1] by dividing by log(K)
    eps = 1e-12
    ent = float(-(p_today * np.log(p_today + eps)).sum())
    ent_norm = float(ent / max(eps, np.log(max(2, n_states))))

    out = {
        "date": str(pd.Timestamp(today_date).date()),
        "state": today_state,
        "regime": labels[today_state],
        "posterior": {labels[i]: float(p_today[i]) for i in range(n_states)},
        "confidence": confidence,
        "uncertainty": uncertainty,
        "entropy": ent,
        "entropy_norm": ent_norm,
    }

    return out


def predict_tomorrow(res: dict, include_expected_returns: bool = True) -> dict[str, Any]:
    """
    One-step ahead regime distribution:
      p_tomorrow = p_today @ A
    """
    wide: pd.DataFrame = res["wide"]
    if wide.empty:
        raise ValueError("wide is empty; cannot predict.")

    model = res["model"]
    A = np.asarray(model.transmat_, dtype=float)

    p_today = _posterior_today(res)
    if A.shape[0] != p_today.shape[0] or A.shape[1] != p_today.shape[0]:
        raise ValueError(f"Transition matrix shape {A.shape} incompatible with p_today shape {p_today.shape}.")

    p_tomorrow = p_today @ A
    p_tomorrow = p_tomorrow / max(1e-12, float(p_tomorrow.sum()))

    n_states = int(p_tomorrow.shape[0])
    labels = _state_labels(res, n_states)

    tomorrow_date = pd.Timestamp(wide.index[-1]) + pd.Timedelta(days=1)
    tomorrow_state = int(np.argmax(p_tomorrow))
    tomorrow_conf = float(np.max(p_tomorrow))

    out = {
        "date": str(pd.Timestamp(tomorrow_date).date()),
        "state": tomorrow_state,
        "regime": labels[tomorrow_state],
        "posterior": {labels[i]: float(p_tomorrow[i]) for i in range(n_states)},
        "confidence": tomorrow_conf,
    }

    if include_expected_returns:
        out["expected_next_day_log_return"] = _expected_next_day_returns(res, p_tomorrow)

    return out


def predict_horizon(res: dict, h: int, include_expected_returns: bool = False) -> dict[str, Any]:
    """
    Multi-step regime forecast:
      p_{t+h} = p_t @ A^h

    - h=0 returns today's posterior
    - h=1 matches predict_tomorrow (without expected returns by default)
    """
    if not isinstance(h, int) or h < 0:
        raise ValueError(f"h must be a non-negative integer, got {h!r}")

    wide: pd.DataFrame = res["wide"]
    if wide.empty:
        raise ValueError("wide is empty; cannot predict.")

    model = res["model"]
    A = np.asarray(model.transmat_, dtype=float)

    p_today = _posterior_today(res)
    k = int(p_today.shape[0])
    if A.shape != (k, k):
        raise ValueError(f"Transition matrix shape {A.shape} incompatible with p_today shape {p_today.shape}.")

    # Matrix power for multi-step transitions
    A_h = np.linalg.matrix_power(A, int(h))
    p_h = p_today @ A_h
    p_h = p_h / max(1e-12, float(p_h.sum()))

    labels = _state_labels(res, k)
    state = int(np.argmax(p_h))
    conf = float(np.max(p_h))

    horizon_date = pd.Timestamp(wide.index[-1]) + pd.Timedelta(days=int(h))
    out = {
        "h": int(h),
        "date": str(pd.Timestamp(horizon_date).date()),
        "state": state,
        "regime": labels[state],
        "posterior": {labels[i]: float(p_h[i]) for i in range(k)},
        "confidence": conf,
    }

    if include_expected_returns:
        out["expected_log_return"] = _expected_next_day_returns(res, p_h)

    return out


