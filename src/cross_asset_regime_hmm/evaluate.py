"""
Evaluation + sanity checks for Cross-Asset Regime HMM runs.

Provides:
- Log-likelihood on training data
- State/regime occupancy
- Average regime duration (run lengths)
- Transition matrix stats (e.g., max off-diagonal)
- Stability across seeds (refit 5x, compare decoded states with ARI)
"""

from __future__ import annotations

import time
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .hmm_model import RegimeHMM


def _best_state_permutation(ref_means: np.ndarray, other_means: np.ndarray) -> list[int]:
    """
    Find permutation p of states that best aligns other->ref by minimizing L2 distance
    between emission means. Returns a list p where p[j] = i means other_state j maps to ref_state i.
    Brute-force permutations are fine for small n_states (<=5 typical here).
    """
    import itertools

    k = int(ref_means.shape[0])
    if other_means.shape[0] != k:
        raise ValueError("State count mismatch during alignment.")

    best_cost = float("inf")
    best_perm = list(range(k))
    for perm in itertools.permutations(range(k)):
        # perm[i] = which other-state aligns to ref-state i
        cost = 0.0
        for ref_i, other_j in enumerate(perm):
            d = ref_means[ref_i] - other_means[other_j]
            cost += float(np.dot(d, d))
        if cost < best_cost:
            best_cost = cost
            best_perm = list(perm)

    # invert to mapping other_state -> ref_state
    inv = [0] * k
    for ref_i, other_j in enumerate(best_perm):
        inv[other_j] = ref_i
    return inv


def _run_lengths(labels: Iterable) -> list[int]:
    labels = list(labels)
    if not labels:
        return []
    runs: list[int] = []
    curr = labels[0]
    k = 1
    for x in labels[1:]:
        if x == curr:
            k += 1
        else:
            runs.append(k)
            curr = x
            k = 1
    runs.append(k)
    return runs


def _get_X_scaled(res: dict) -> np.ndarray:
    wide = res["wide"]
    feature_cols = res["wide_feature_cols"]
    scaler = res["scaler"]
    X = wide[feature_cols].to_numpy()
    return scaler.transform(X)


def _get_model_ll(model: RegimeHMM, X_scaled: np.ndarray) -> float:
    # hmmlearn GaussianHMM supports .score(X) => total log likelihood
    if hasattr(model, "model") and hasattr(model.model, "score"):
        return float(model.model.score(X_scaled))
    if hasattr(model, "score"):
        return float(model.score(X_scaled))
    raise AttributeError("Model does not expose a score() method for log-likelihood.")


def evaluate_run(
    res: dict,
    n_refits: int = 5,
    seeds: Optional[list[int]] = None,
) -> None:
    """
    Print a compact evaluation report for a pipeline run result `res`.

    `res` is expected to be the output of `run_pipeline()`:
      - res['wide'], res['wide_feature_cols'], res['scaler'], res['model'], res['label_map']
    """
    wide: pd.DataFrame = res["wide"]
    model: RegimeHMM = res["model"]
    label_map: dict = res.get("label_map", {})

    X_scaled = _get_X_scaled(res)
    n = int(X_scaled.shape[0])

    # Log-likelihood
    ll_total = _get_model_ll(model, X_scaled)
    ll_per_step = ll_total / max(1, n)

    # Occupancy
    if "regime" in wide.columns and wide["regime"].notna().any():
        occ = wide["regime"].value_counts(normalize=True).sort_index()
        occ_label = "Regime"
    else:
        occ = wide["state"].value_counts(normalize=True).sort_index()
        occ_label = "State"

    # Durations
    seq_for_runs = wide["regime"].tolist() if "regime" in wide.columns else wide["state"].tolist()
    runs = _run_lengths(seq_for_runs)
    avg_run = float(np.mean(runs)) if runs else float("nan")
    med_run = float(np.median(runs)) if runs else float("nan")

    # Transition matrix stats
    A = np.array(model.transmat_, dtype=float)
    diag_mean = float(np.mean(np.diag(A))) if A.size else float("nan")
    off = A.copy()
    if off.size:
        np.fill_diagonal(off, -np.inf)
    max_offdiag = float(np.max(off)) if off.size else float("nan")

    # Stability across seeds
    if seeds is None:
        # choose stable, deterministic set
        seeds = [1, 2, 3, 4, 5][: max(0, int(n_refits))]
    else:
        seeds = seeds[: max(0, int(n_refits))]

    base_states = wide["state"].to_numpy()
    ari_scores: list[float] = []
    ari_aligned_scores: list[float] = []
    ll_refits: list[float] = []
    cov_type = getattr(model.model, "covariance_type", "diag")
    n_iter = int(getattr(model.model, "n_iter", 200))
    n_states = int(getattr(model.model, "n_components", 3))

    t0 = time.time()
    for s in seeds:
        m = RegimeHMM(n_states=n_states, covariance_type=cov_type, n_iter=n_iter, random_state=int(s))
        m.fit(X_scaled)
        ll_refits.append(_get_model_ll(m, X_scaled))
        st = m.predict_states(X_scaled)
        ari_scores.append(float(adjusted_rand_score(base_states, st)))

        # Align state IDs before ARI (HMM state labels are permutation-invariant)
        p = _best_state_permutation(model.means_, m.means_)
        st_aligned = np.vectorize(lambda x: p[int(x)])(st)
        ari_aligned_scores.append(float(adjusted_rand_score(base_states, st_aligned)))
    dt = float(time.time() - t0)

    # Print compact report
    print("\n" + "=" * 72)
    print("Cross-Asset Regime HMM — Evaluation Report")
    print("=" * 72)
    print(f"- Log-likelihood (total): {ll_total:,.2f}")
    print(f"- Log-likelihood (per step): {ll_per_step:,.4f}")
    print("")
    print(f"- {occ_label} occupancy (% of time):")
    for k, v in occ.items():
        print(f"  - {k}: {100.0 * float(v):.1f}%")
    print("")
    print(f"- Average regime duration (run length): {avg_run:.2f} days (median {med_run:.0f})")
    print(f"- Transition matrix: mean(diagonal)={diag_mean:.3f}, max(off-diagonal)={max_offdiag:.3f}")
    if ari_scores:
        print(
            f"- Stability across seeds (ARI vs baseline, n={len(ari_scores)}): "
            f"raw mean={np.mean(ari_scores):.3f}, aligned mean={np.mean(ari_aligned_scores):.3f} "
            f"(raw min={np.min(ari_scores):.3f}, aligned min={np.min(ari_aligned_scores):.3f}) "
            f"(~{dt:.1f}s)"
        )
        if ll_refits:
            print(
                f"- Refit restarts (log-likelihood): "
                f"best={np.max(ll_refits):,.2f}, median={np.median(ll_refits):,.2f}, worst={np.min(ll_refits):,.2f}"
            )
    print("=" * 72 + "\n")


