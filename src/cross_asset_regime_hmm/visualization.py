"""
Visualization module for Cross-Asset Regime HMM.

Each plot answers one of two questions:
1. What regime are we in, and when did it change?
2. What does each regime mean in the real world of prices/returns/vol?
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out.index = pd.to_datetime(out.index)
        return out
    return df

# Regime color palette - distinct, colorblind-friendly
REGIME_COLORS = {
    "Bull": "#2ecc71",       # green
    "Bear": "#e74c3c",       # red
    "Volatile": "#f39c12",   # orange
    "Calm": "#3498db",       # blue
    "Unknown": "#95a5a6",    # gray
}

def _get_regime_color(regime: str) -> str:
    """Get color for regime, with fallback."""
    return REGIME_COLORS.get(regime, REGIME_COLORS["Unknown"])


def _extract_symbols(res: dict) -> list[str]:
    """Extract symbol list from pipeline result."""
    return list(res["per_asset"].keys())


# =============================================================================
# 1) Regime Timeline with Posterior Confidence
# =============================================================================

def plot_regime_timeline(
    res: dict,
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Two-panel plot:
    - Top: Regime strip (solid colors showing which regime)
    - Bottom: Confidence line showing how certain the model is
    """
    wide = _ensure_datetime_index(res["wide"])
    model = res["model"]
    scaler = res["scaler"]
    feature_cols = res["wide_feature_cols"]

    # Get posterior probabilities
    X_scaled = scaler.transform(wide[feature_cols].to_numpy())
    probs = model.predict_probs(X_scaled)  # (n_days, n_states)
    confidence = probs.max(axis=1)

    dates = wide.index
    regimes = wide["regime"].values
    if len(dates) == 0:
        raise ValueError("plot_regime_timeline: wide dataframe has no rows.")

    end_span = dates[-1] + pd.Timedelta(days=1)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                              gridspec_kw={"height_ratios": [1, 2]})

    # --- Top panel: Regime strip ---
    ax_regime = axes[0]
    prev_regime = None
    start_idx = 0
    for i, regime in enumerate(regimes):
        if regime != prev_regime and prev_regime is not None:
            ax_regime.axvspan(dates[start_idx], dates[i], color=_get_regime_color(prev_regime), alpha=0.9)
            start_idx = i
        prev_regime = regime
    # Final segment
    ax_regime.axvspan(dates[start_idx], end_span, color=_get_regime_color(prev_regime), alpha=0.9)

    ax_regime.set_yticks([])
    ax_regime.set_ylabel("Regime")
    ax_regime.set_title("Regime Timeline with Posterior Confidence")

    # Legend for regimes
    unique_regimes = wide["regime"].dropna().unique()
    patches = [mpatches.Patch(color=_get_regime_color(r), label=r) for r in sorted(unique_regimes)]
    ax_regime.legend(handles=patches, loc="upper left", framealpha=0.9, ncol=max(1, len(unique_regimes)))

    # --- Bottom panel: Confidence line ---
    ax_conf = axes[1]
    
    # Color the confidence line by regime
    for i in range(len(dates) - 1):
        ax_conf.plot(
            [dates[i], dates[i + 1]], 
            [confidence[i], confidence[i + 1]], 
            color=_get_regime_color(regimes[i]), 
            linewidth=1.5
        )
    
    # Add a filled area to make it more visible
    ax_conf.fill_between(dates, confidence, alpha=0.3, color="#3498db")
    
    # Reference lines
    ax_conf.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="50% confidence")
    ax_conf.axhline(0.8, color="gray", linestyle=":", linewidth=0.8, label="80% confidence")

    ax_conf.set_xlim(dates.min(), end_span)
    ax_conf.set_ylim(0, 1.05)
    ax_conf.set_ylabel("Confidence (max prob)")
    ax_conf.set_xlabel("Date")
    ax_conf.legend(loc="lower right", framealpha=0.9)

    # Add confidence stats as text
    avg_conf = confidence.mean()
    min_conf = confidence.min()
    ax_conf.text(0.02, 0.12, f"Avg: {avg_conf:.1%}  Min: {min_conf:.1%}", 
                 transform=ax_conf.transAxes, fontsize=9, 
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# =============================================================================
# 2) Price Charts with Regime Shading (per asset)
# =============================================================================

def plot_price_with_regimes(
    res: dict,
    symbols: Optional[list[str]] = None,
    figsize: tuple = (14, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    For each asset: line plot of price with background shaded by inferred regime.
    This is the 'sniff test' plot - does the regime make sense visually?
    """
    wide = _ensure_datetime_index(res["wide"])
    per_asset = res["per_asset"]

    if symbols is None:
        symbols = _extract_symbols(res)

    n_assets = len(symbols)
    fig, axes = plt.subplots(n_assets, 1, figsize=(figsize[0], figsize[1] * n_assets), sharex=True)
    if n_assets == 1:
        axes = [axes]

    dates = wide.index
    regimes = wide["regime"].values
    if len(dates) == 0:
        raise ValueError("plot_price_with_regimes: wide dataframe has no rows.")
    end_span = dates[-1] + pd.Timedelta(days=1)

    for ax, sym in zip(axes, symbols):
        # Get price data aligned to common dates
        df_sym = _ensure_datetime_index(per_asset[sym]["data"])
        price = df_sym.reindex(dates)["price"].to_numpy()

        # Plot price line
        ax.plot(dates, price, color="#2c3e50", linewidth=1, label=sym)

        # Shade background by regime
        prev_regime = None
        start_idx = 0
        for i, regime in enumerate(regimes):
            if regime != prev_regime and prev_regime is not None:
                ax.axvspan(dates[start_idx], dates[i], alpha=0.3, color=_get_regime_color(prev_regime), linewidth=0)
                start_idx = i
            prev_regime = regime
        # Final segment
        ax.axvspan(dates[start_idx], end_span, alpha=0.3, color=_get_regime_color(prev_regime), linewidth=0)

        ax.set_ylabel(f"{sym} Price")
        ax.set_title(f"{sym} Price with Regime Shading")
        ax.legend(loc="upper left")

    axes[-1].set_xlabel("Date")

    # Shared legend for regimes
    unique_regimes = wide["regime"].dropna().unique()
    patches = [mpatches.Patch(color=_get_regime_color(r), alpha=0.3, label=r) for r in sorted(unique_regimes)]
    fig.legend(handles=patches, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# =============================================================================
# 3) Return + Volatility Summary per Regime (box/violin)
# =============================================================================

def plot_regime_distributions(
    res: dict,
    symbols: Optional[list[str]] = None,
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Per regime: distribution of returns and volatility.
    This is where regime labels become statistical personalities.
    """
    wide = res["wide"]

    if symbols is None:
        symbols = _extract_symbols(res)

    # Build long-form data for seaborn
    records = []
    for sym in symbols:
        ret_col = f"{sym}_log_return"
        vol_col = f"{sym}_roll_vol"
        for idx, row in wide.iterrows():
            records.append({
                "Symbol": sym,
                "Regime": row["regime"],
                "Log Return": row[ret_col],
                "Rolling Vol": row[vol_col],
            })
    df_long = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Returns distribution
    palette = {r: _get_regime_color(r) for r in df_long["Regime"].unique()}
    sns.boxplot(data=df_long, x="Regime", y="Log Return", hue="Symbol", ax=axes[0], palette="Set2")
    axes[0].set_title("Log Return Distribution by Regime")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)

    # Volatility distribution
    sns.boxplot(data=df_long, x="Regime", y="Rolling Vol", hue="Symbol", ax=axes[1], palette="Set2")
    axes[1].set_title("Rolling Volatility Distribution by Regime")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_aggregate_regime_distributions(
    res: dict,
    symbols: Optional[list[str]] = None,
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Aggregate return and vol across all assets, per regime.
    Violin plots show the full distribution shape.
    """
    wide = res["wide"]

    if symbols is None:
        symbols = _extract_symbols(res)

    # Compute aggregate metrics
    ret_cols = [f"{s}_log_return" for s in symbols]
    vol_cols = [f"{s}_roll_vol" for s in symbols]

    df = wide.copy()
    df["avg_return"] = df[ret_cols].mean(axis=1)
    df["avg_vol"] = df[vol_cols].mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    palette = {r: _get_regime_color(r) for r in df["regime"].unique()}

    sns.violinplot(
        data=df,
        x="regime",
        y="avg_return",
        hue="regime",
        palette=palette,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Aggregate Return Distribution by Regime")
    axes[0].set_xlabel("Regime")
    axes[0].set_ylabel("Mean Log Return (across assets)")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)

    sns.violinplot(
        data=df,
        x="regime",
        y="avg_vol",
        hue="regime",
        palette=palette,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Aggregate Volatility Distribution by Regime")
    axes[1].set_xlabel("Regime")
    axes[1].set_ylabel("Mean Rolling Vol (across assets)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# =============================================================================
# 4) Transition Matrix Heatmap (labeled)
# =============================================================================

def plot_transition_matrix(
    res: dict,
    figsize: tuple = (7, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of transition matrix A, with rows/cols labeled by regime names.
    Shows regime stickiness (high diagonal) and common transitions.
    """
    model = res["model"]
    label_map = res["label_map"]

    transmat = model.transmat_
    n_states = transmat.shape[0]

    # Order labels by state index
    labels = [label_map.get(i, f"State {i}") for i in range(n_states)]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        transmat,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Transition Probability"},
    )
    ax.set_xlabel("To Regime")
    ax.set_ylabel("From Regime")
    ax.set_title("Regime Transition Matrix")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# =============================================================================
# 5) State Occupancy + Average Duration
# =============================================================================

def plot_state_occupancy(
    res: dict,
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart: % of time spent in each regime + average run length.
    If a regime lasts only 1-2 days on average, it's likely noise.
    """
    wide = res["wide"]
    label_map = res["label_map"]

    states = wide["state"].values
    regimes = wide["regime"].values

    # Occupancy: % of time in each regime
    regime_counts = pd.Series(regimes).value_counts(normalize=True).sort_index()

    # Average run length (consecutive days in same regime)
    run_lengths = {r: [] for r in regime_counts.index}
    current_regime = regimes[0]
    current_run = 1

    for i in range(1, len(regimes)):
        if regimes[i] == current_regime:
            current_run += 1
        else:
            run_lengths[current_regime].append(current_run)
            current_regime = regimes[i]
            current_run = 1
    run_lengths[current_regime].append(current_run)  # final run

    avg_run_length = {r: np.mean(lengths) if lengths else 0 for r, lengths in run_lengths.items()}

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Occupancy bar chart
    colors = [_get_regime_color(r) for r in regime_counts.index]
    axes[0].bar(regime_counts.index, regime_counts.values * 100, color=colors, edgecolor="black")
    axes[0].set_ylabel("% of Time")
    axes[0].set_xlabel("Regime")
    axes[0].set_title("Regime Occupancy (% of Days)")
    axes[0].set_ylim(0, 100)

    # Average run length bar chart
    regimes_sorted = list(avg_run_length.keys())
    colors = [_get_regime_color(r) for r in regimes_sorted]
    axes[1].bar(regimes_sorted, [avg_run_length[r] for r in regimes_sorted], color=colors, edgecolor="black")
    axes[1].set_ylabel("Average Days")
    axes[1].set_xlabel("Regime")
    axes[1].set_title("Average Regime Duration (Run Length)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# =============================================================================
# 6) Cross-Asset Coherence Plot
# =============================================================================

def plot_cross_asset_coherence(
    res: dict,
    symbols: Optional[list[str]] = None,
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Show cross-asset coherence: how many assets are up vs down per day, colored by regime.
    Also shows mean return across assets per day.
    Reveals whether the 'global state' corresponds to a coherent market mode.
    """
    wide = res["wide"]

    if symbols is None:
        symbols = _extract_symbols(res)

    ret_cols = [f"{s}_log_return" for s in symbols]

    df = wide.copy()
    # Count assets with positive returns each day
    df["n_up"] = (df[ret_cols] > 0).sum(axis=1)
    df["n_down"] = (df[ret_cols] < 0).sum(axis=1)
    df["mean_return"] = df[ret_cols].mean(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    dates = df.index
    regimes = df["regime"].values

    # Plot 1: Stacked bar of up/down assets, colored by regime for background
    ax1 = axes[0]
    for i, (date, regime) in enumerate(zip(dates, regimes)):
        ax1.axvspan(date, date + pd.Timedelta(days=1), alpha=0.2, color=_get_regime_color(regime), linewidth=0)

    ax1.bar(dates, df["n_up"], color="#2ecc71", label="Assets Up", alpha=0.8, width=1)
    ax1.bar(dates, -df["n_down"], color="#e74c3c", label="Assets Down", alpha=0.8, width=1)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("# Assets Up / Down")
    ax1.set_title("Cross-Asset Coherence: Assets Up vs Down (background = regime)")
    ax1.legend(loc="upper left")

    # Plot 2: Mean return across assets, colored by regime
    ax2 = axes[1]
    for regime in df["regime"].unique():
        mask = df["regime"] == regime
        ax2.scatter(dates[mask], df.loc[mask, "mean_return"], c=_get_regime_color(regime), label=regime, alpha=0.6, s=10)

    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Mean Log Return")
    ax2.set_xlabel("Date")
    ax2.set_title("Mean Return Across Assets (colored by regime)")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# =============================================================================
# Master function: Generate all plots
# =============================================================================

def generate_all_plots(
    res: dict,
    output_dir: Optional[str] = None,
    symbols: Optional[list[str]] = None,
    show: bool = True,
) -> dict:
    """
    Generate all visualization plots and optionally save to output_dir.

    Returns dict of figure objects.
    """
    # If we are not showing figures, force non-interactive plotting to avoid any GUI popups.
    # (Some environments may still display windows on figure creation unless we disable it.)
    if not show:
        try:
            plt.ioff()
            plt.switch_backend("Agg")
        except Exception:
            pass

    save = output_dir is not None
    output_path = Path(output_dir) if save else None
    if save:
        output_path.mkdir(parents=True, exist_ok=True)

    if symbols is None:
        symbols = _extract_symbols(res)

    figures = {}

    print("Generating regime visualizations...")

    # 1) Regime timeline
    figures["timeline"] = plot_regime_timeline(
        res, save_path=(str(output_path / "01_regime_timeline.png") if save else None)
    )
    print("  - Regime timeline with confidence")

    # 2) Price with regime shading
    figures["price_regimes"] = plot_price_with_regimes(
        res, symbols=symbols, save_path=(str(output_path / "02_price_with_regimes.png") if save else None)
    )
    print("  - Price charts with regime shading")

    # 3a) Per-asset distributions
    figures["distributions"] = plot_regime_distributions(
        res, symbols=symbols, save_path=(str(output_path / "03a_regime_distributions.png") if save else None)
    )
    print("  - Return/vol distributions by regime (per asset)")

    # 3b) Aggregate distributions
    figures["aggregate_distributions"] = plot_aggregate_regime_distributions(
        res, symbols=symbols, save_path=(str(output_path / "03b_aggregate_distributions.png") if save else None)
    )
    print("  - Aggregate return/vol distributions by regime")

    # 4) Transition matrix
    figures["transition_matrix"] = plot_transition_matrix(
        res, save_path=(str(output_path / "04_transition_matrix.png") if save else None)
    )
    print("  - Transition matrix heatmap")

    # 5) State occupancy
    figures["occupancy"] = plot_state_occupancy(
        res, save_path=(str(output_path / "05_state_occupancy.png") if save else None)
    )
    print("  - State occupancy & duration")

    # 6) Cross-asset coherence
    figures["coherence"] = plot_cross_asset_coherence(
        res, symbols=symbols, save_path=(str(output_path / "06_cross_asset_coherence.png") if save else None)
    )
    print("  - Cross-asset coherence plot")

    if save:
        print(f"\nAll plots saved to: {output_path.resolve()}")

    if show:
        plt.show()
    else:
        try:
            plt.close("all")
        except Exception:
            pass

    return figures

