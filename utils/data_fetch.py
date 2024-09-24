import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from fredapi import Fred
import json

def load_config():
    with open('config.json') as config_file:
        return json.load(config_file)

# TODO: flexible date range
def fetch_adj_close_prices(tickers):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 10)

    adj_close_df = pd.DataFrame()
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        if 'Adj Close' in data.columns:
            adj_close_df[ticker] = data['Adj Close']
        else:
            raise ValueError(f"Data for {ticker} not available or incomplete.")
    
    if adj_close_df.empty:
        raise ValueError("No valid data found for the provided tickers.")

    return adj_close_df

def fetch_risk_free_rate():
    config = load_config()
    fred = Fred(api_key=config['FRED_API_KEY'])

    try:
        ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
        return ten_year_treasury_rate.iloc[-1]
    except Exception as e:
        raise RuntimeError(f"Error fetching risk-free rate: {e}")
