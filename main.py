from cross_asset_regime_hmm import stocks, crypto, run_pipeline

def main():
    start = ""
    end = ""
    
    res = run_pipeline(stocks + crypto, start, end)
    print(f"Result: {res}")

if __name__ == "__main__":
    main()