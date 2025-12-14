from src.cross_asset_regime_hmm import stocks, crypto, run_pipeline

def main():
    start = "2025-01-01"
    end = "2025-02-13"
    
    res = run_pipeline(stocks + crypto, start, end)
    print(f"Result: {res}")

if __name__ == "__main__":
    main()