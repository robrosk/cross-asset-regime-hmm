import numpy as np

def label_states_cross_asset(
    means: np.ndarray,
    n_states: int,
    return_indices: list[int],
    vol_indices: list[int],
) -> dict[int, str]:
    """
    Label states using aggregate return/vol across assets.
    - Bull: highest mean aggregate return
    - Bear: lowest mean aggregate return
    - Volatile: highest mean aggregate vol among remaining states
    Remaining states get State{k}.
    """
    # aggregate signals per state
    ret_mu = means[:, return_indices].mean(axis=1)
    vol_mu = means[:, vol_indices].mean(axis=1)

    bull = int(ret_mu.argmax())
    bear = int(ret_mu.argmin())

    remaining = [s for s in range(n_states) if s not in {bull, bear}]
    label_map = {bull: "Bull", bear: "Bear"}

    if remaining:
        volatile = int(remaining[np.argmax(vol_mu[remaining])])
        label_map[volatile] = "Volatile"

    for s in range(n_states):
        label_map.setdefault(s, f"State{s}")

    return label_map


