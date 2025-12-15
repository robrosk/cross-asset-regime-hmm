import argparse
from datetime import datetime, timedelta
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Asset Regime HMM: Train on historical data and predict next-day regime."
    )

    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to include (space-separated), e.g. --symbols SPY QQQ BTC-USD",
    )
    parser.add_argument(
        "--vol-window",
        type=int,
        default=30,
        help="Rolling volatility window length (days). Controls how many past days are used for rolling volatility.",
    )
    parser.add_argument(
        "--covariance-type",
        default="full",
        choices=["full", "diag", "tied", "spherical"],
        help="Gaussian HMM covariance type (hmmlearn). Controls how flexible the emission covariance is.",
    )
    parser.add_argument("--plot", action="store_true", help="Generate visualization plots")
    parser.add_argument(
        "--plot-dir",
        action="store_true",
        help="Save plots to ./plots (fixed; no custom path)",
    )

    args = parser.parse_args()

    show = bool(args.plot)
    output_dir = "plots" if args.plot_dir else None

    if args.plot_dir and not args.plot:
        os.environ.setdefault("MPLBACKEND", "Agg")

    from src.cross_asset_regime_hmm import (
        stocks,
        crypto,
        run_pipeline,
        run_pipeline_walk_forward,
        print_predictions,
        generate_all_plots,
        evaluate_run,
    )

    # Parse end date to show prediction target
    _ = datetime.strptime(args.end, "%Y-%m-%d")  # validate format
    print(f"\nRequested window: {args.start} -> {args.end}")

    symbols = args.symbols if args.symbols is not None else (stocks + crypto)
    
    print(f"Symbols: {symbols}\n")
    print(f"vol_window: {args.vol_window}\n")
    print(f"covariance_type: {args.covariance_type}\n")

    res = run_pipeline(
        symbols=symbols,
        start=args.start,
        end=args.end,
        vol_window=args.vol_window,
        n_states=3,
        covariance_type=args.covariance_type,
    )

    evaluate_run(res=res)

    train_end = str(pd.Timestamp(res["wide"].index[-1]).date())
    print(f"\nTraining data ends on: {train_end}\n")

    print_predictions(res, top_k=5)

    if args.plot or args.plot_dir:
        print("\n" + "=" * 60)
        generate_all_plots(res, output_dir=output_dir, show=show)

if __name__ == "__main__":
    main()