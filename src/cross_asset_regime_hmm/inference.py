import numpy as np
import pandas as pd

from .predict import predict_today, predict_tomorrow, predict_horizon

class RegimeInference:
    """
    Small API wrapper around a pipeline run result.

    Supports:
      - predict_today() -> dict
      - predict_tomorrow() -> dict
    """

    def __init__(self, res: dict):
        self.res = res

    def predict_today(self) -> dict:
        return predict_today(self.res)

    def predict_tomorrow(self, include_expected_returns: bool = True) -> dict:
        return predict_tomorrow(self.res, include_expected_returns=include_expected_returns)

    def predict_horizon(self, h: int, include_expected_returns: bool = False) -> dict:
        """
        Predict h days ahead using p_{t+h} = p_t @ A^h.
        """
        return predict_horizon(self.res, h=h, include_expected_returns=include_expected_returns)


def print_predictions(
    res: dict,
    *,
    top_k: int = 5,
    horizons_days: list[int] | None = None,
) -> None:
    """
    CLI print helper.

    Prints:
      - today (h=0)
      - tomorrow (h=1)
      - next week (h=7)
      - next month (h=30)
      - then once a month for the rest of the year (every 30 days up to 360)
    """
    inf = RegimeInference(res)
    if horizons_days is None:
        horizons_days = [0, 1, 7, 30] + list(range(60, 361, 30))

    print("\n=== Regime forecast ===")
    for h in horizons_days:
        include_rets = (h == 1)  # only print expected returns for tomorrow by default
        pred = inf.predict_horizon(h=h, include_expected_returns=include_rets)

        label = (
            "Today" if h == 0 else
            "Tomorrow" if h == 1 else
            "Next week" if h == 7 else
            "Next month" if h == 30 else
            f"+{h}d"
        )
        conf = pred.get("confidence", float("nan"))
        print(f"- {label:<10s} ({pred['date']}): {pred['regime']}  (confidence={conf:.3f})")

        if include_rets:
            exp = pred.get("expected_log_return") or {}
            if exp:
                rows = sorted(exp.items(), key=lambda x: abs(x[1]), reverse=True)[: max(1, int(top_k))]
                print("  expected next-day log return (top by |value|):")
                for sym, r in rows:
                    print(f"  - {sym}: {r:+.6f}")


def print_today_tomorrow_predictions(res: dict, top_k: int = None) -> None:
    """
    Backwards-compatible alias.
    """
    print_predictions(res, top_k=(5 if top_k is None else top_k), horizons_days=[0, 1])
