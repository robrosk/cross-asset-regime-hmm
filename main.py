import argparse
from datetime import datetime, timedelta
import os

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Asset Regime HMM: Train on historical data and predict next-day regime."
    )

    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
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
        print_today_tomorrow_predictions,
        generate_all_plots,
        evaluate_run,
    )

    # Parse end date to show prediction target
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    predict_dt = end_dt + timedelta(days=1)

    print(f"\nTraining on data from {args.start} to {args.end}")
    print(f"Predicting regime for: {predict_dt.strftime('%Y-%m-%d')}\n")

    res = run_pipeline(symbols=stocks + crypto, start=args.start, end=args.end, vol_window=50, n_states=3)

    evaluate_run(res=res)

    print_today_tomorrow_predictions(res=res)

    if args.plot or args.plot_dir:
        print("\n" + "=" * 60)
        generate_all_plots(res, output_dir=output_dir, show=show)

if __name__ == "__main__":
    main()