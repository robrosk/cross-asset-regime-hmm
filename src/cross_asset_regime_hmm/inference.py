import numpy as np
import pandas as pd

def print_today_tomorrow_predictions(res: dict, top_k: int = None) -> None:
    wide: pd.DataFrame = res["wide"]
    model = res["model"]
    scaler = res["scaler"]
    label_map: dict[int, str] = res["label_map"]
    wide_feature_cols: list[str] = res.get("wide_feature_cols", [])

    if wide.empty:
        raise ValueError("wide is empty; cannot compute predictions.")

    # Posterior regime probabilities for each day (filtering)
    post = model.predict_probs(scaler.transform(wide[wide_feature_cols].to_numpy()))
    p_today = post[-1]  # shape (n_states,)

    # Transition matrix
    A = model.transmat_  # shape (n_states, n_states)

    # One-step-ahead regime distribution
    p_tomorrow = p_today @ A

    # Map state->label for printing (ensure order 0..n_states-1)
    n_states = len(p_today)
    labels = [label_map.get(i, f"State{i}") for i in range(n_states)]

    today_date = wide.index[-1]
    print(f"\n=== Regime probabilities for today ({today_date.date()}) ===")
    for i, lab in enumerate(labels):
        print(f"{lab:>10s} (state {i}): {p_today[i]:.4f}")

    today_state = int(np.argmax(p_today))
    print(f"Most likely regime today: {labels[today_state]} (state {today_state})\n")

    print("=== Regime probabilities for tomorrow (1-step ahead) ===")
    for i, lab in enumerate(labels):
        print(f"{lab:>10s} (state {i}): {p_tomorrow[i]:.4f}")

    tomorrow_state = int(np.argmax(p_tomorrow))
    print(f"Most likely regime tomorrow: {labels[tomorrow_state]} (state {tomorrow_state})\n")

    # Optional: Expected per-asset returns tomorrow (mixture of emission means)
    # This assumes your feature layout is [sym1_ret, sym1_vol, sym2_ret, sym2_vol, ...]
    if wide_feature_cols:
        # return features are at even indices: 0,2,4,...
        return_indices = list(range(0, len(wide_feature_cols), 2))

        mu_scaled = model.means_[:, return_indices]              # (n_states, n_assets)
        E_ret_scaled = p_tomorrow @ mu_scaled                    # (n_assets,)

        means = scaler.mean_[return_indices]
        scales = scaler.scale_[return_indices]
        E_ret = E_ret_scaled * scales + means                    # back to original units

        sym_names = [col.replace("_log_return", "") for col in wide_feature_cols[::2]]

        print("=== Expected next-day log return (mixture of regime means) ===")
        rows = list(zip(sym_names, E_ret))
        if top_k is not None:
            # show biggest magnitude predictions
            rows = sorted(rows, key=lambda x: abs(x[1]), reverse=True)[:top_k]

        for sym, r in rows:
            print(f"{sym:>10s}: {r:+.6f}")
        print()
