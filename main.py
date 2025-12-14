from src.cross_asset_regime_hmm import stocks, crypto, run_pipeline, print_today_tomorrow_predictions

def main():
    start = "2023-01-01"
    end = "2025-12-13"
    
    res = run_pipeline(symbols=stocks + crypto, start=start, end=end)
    print_today_tomorrow_predictions(res=res)

    

if __name__ == "__main__":
    main()