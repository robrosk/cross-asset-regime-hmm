# Cross-Asset Regime HMM (Hidden Markov Model)

This project builds a **cross-asset market regime detector** using a **Gaussian Hidden Markov Model (HMM)** over an *observation vector* made from multiple assets (e.g., equities + crypto).

At a high level:
- Pull OHLCV data (currently via `yfinance`)
- Engineer simple features per asset (currently: **log returns** + **rolling volatility**)
- Align all assets by **common dates** (intersection)
- Fit an HMM to the **cross-asset feature matrix**
- Produce:
  - Regime labels (e.g. Bull/Bear/Volatile)
  - A daily regime report (today / tomorrow / forward horizons)
  - Evaluation metrics (likelihood, occupancy, run-lengths, transition stats, stability)
  - Optional plots

## Status: Under Development

This repo is actively evolving: features, labeling, and model-selection logic will change as the project matures.

## What the HMM is doing (in plain English)

An HMM assumes there is a **hidden regime/state** (e.g., “Bull”, “Bear”, “Volatile”) that you don’t directly observe.  
What you *do* observe each day is a feature vector (here: **log returns** and **rolling volatility** across assets).

The model learns:
- A **transition matrix** \(A\): how likely regimes are to persist or switch (e.g., “Volatile” → “Volatile”)
- **Emission distributions**: for each regime, what returns/volatility “typically look like”

Forecasts are usually dominated by **regime persistence** unless the observed features strongly suggest a switch.

## Critical Disclaimer (Please Read)

This is a **Hidden Markov Model** trained purely on **historical market evidence** (returns/volatility). It:
- **Cannot see “new information”** outside the historical feature stream (news, macro events, structural breaks).
- **Cannot inherently “track trend shifts”** unless they appear in the data in a way the features capture.
- Will often predict **persistence** because HMM transition matrices tend to be “sticky” (high diagonal).

In practice, the most reliable “predictions” are typically **short-horizon regime persistence forecasts**.  
**Expect the best accuracy within ~1 week of the current regime**, and treat longer-horizon outputs as *scenario projections* rather than sharp forecasts.

## Installation

Create/activate a Python environment (recommended) and install dependencies:

```bash
pip install -r requirements.txt
```

## CLI Usage

The main entrypoint is `main.py`. You must provide a start and end date:

```bash
python main.py --start 2021-01-01 --end 2025-12-14
```

### Choose symbols from the CLI

By default the CLI uses the repo’s configured symbol lists (`stocks + crypto`).  
To override from the command line, pass `--symbols` (space-separated):

```bash
python main.py --start 2021-01-01 --end 2025-12-14 --symbols SPY QQQ BTC-USD ETH-USD
```

Tip: symbols like `BTC-USD` are fine unquoted in most shells, but quoting is also safe:

```bash
python main.py --start 2021-01-01 --end 2025-12-14 --symbols "BTC-USD" "ETH-USD"
```

What it does:
- Trains the model on the aligned dataset ending at the last available common date
- Prints an evaluation report
- Prints a regime forecast summary:
  - Today, tomorrow, next week, next month, then monthly horizons out to ~1 year

### `--vol-window` (rolling volatility window)

We model each day using two features: **daily log return** and **rolling volatility**.  
`--vol-window` controls how many past days are used to compute rolling volatility:

- **Smaller window** (e.g., **5–10 days**) → volatility reacts faster to recent moves
  - Detects shocks quickly
  - Regimes may switch more often (more “choppy” state timeline)

- **Medium window** (e.g., **20 days**) → balanced, common default
  - Captures “monthly-ish” volatility shifts
  - Typically smoother regimes than 5–10

- **Larger window** (e.g., **40–60+ days**) → volatility changes slowly
  - Regimes tend to be more persistent
  - Turning points can lag (volatility stays elevated after markets calm)

Rule of thumb: pick a window that matches the timescale you want “regime” to mean (**weekly vs monthly vs quarterly**).

Example:

```bash
python main.py --start 2021-01-01 --end 2025-12-14 --symbols SPY QQQ BTC-USD --vol-window 20
```

### `--covariance-type` (HMM emission covariance)

The HMM uses Gaussian emissions. `--covariance-type` controls the **shape/flexibility** of the covariance matrix used to model the feature vector inside each regime:

- **`full`**: each regime has a full covariance matrix (features can be correlated)
  - Most flexible, but easiest to overfit with limited data

- **`diag`**: each regime has a diagonal covariance (features treated as independent given the regime)
  - Often more stable; good default when you have many features or noisy data

- **`tied`**: all regimes share one full covariance matrix
  - Can be a good compromise: correlated features, fewer parameters than `full`

- **`spherical`**: each regime has a single variance value shared across all features
  - Strong regularization; often too restrictive but can help with stability

Example:

```bash
python main.py --start 2021-01-01 --end 2025-12-14 --symbols SPY QQQ BTC-USD --vol-window 20 --covariance-type diag
```

### Plotting

- **Show plots (no saving):**

```bash
python main.py --start 2021-01-01 --end 2025-12-14 --plot
```

- **Save plots to `./plots` (no GUI windows):**

```bash
python main.py --start 2021-01-01 --end 2025-12-14 --plot-dir
```

- **Show + save:**

```bash
python main.py --start 2021-01-01 --end 2025-12-14 --plot --plot-dir
```

## Outputs

- **Evaluation report**: log-likelihood, state/regime occupancy, average duration (run-lengths), transition stats, and stability across random restarts.
- **Regime forecasts**: multi-step forecasts computed as \(p_{t+h} = p_t A^h\).
- **Plots** (optional): regime timeline with confidence, per-asset price shading, regime distributions, transition matrix, occupancy/duration, and cross-asset coherence.

## Notes

- Data availability differs by asset (weekends/holidays vs 24/7 crypto). The pipeline currently takes the **intersection** of dates across all assets to keep the cross-asset matrix aligned.
