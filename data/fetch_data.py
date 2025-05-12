import yfinance as yf
import pandas as pd

def fetch_and_save(tickers='BZ=F',
               period='10y'):
    df = yf.download(tickers, period=period)
    df[['Close']].to_csv('data/oil_prices.csv')

if __name__ == "__main__":
    fetch_and_save()
    print("Data fetched and saved to data/oil_prices.csv")