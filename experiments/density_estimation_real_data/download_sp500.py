import pandas as pd
import requests
import yfinance as yf
from io import StringIO

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Get current S&P 500 constituents from Wikipedia
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)
response.raise_for_status()

tables = pd.read_html(StringIO(response.text))
tickers = tables[0]["Symbol"].tolist()

# Yahoo Finance uses "-" instead of "." in tickers, e.g. BRK.B -> BRK-B
tickers = [ticker.replace(".", "-") for ticker in tickers]

all_data = []

for batch in chunks(tickers, 50):
    print(f"Downloading batch with {len(batch)} tickers...")

    data = yf.download(
        batch,
        period="13y",
        interval="1d",
        auto_adjust=False,
        progress=True,
        threads=True
    )

    close = data["Close"] # type: ignore
    volume = data["Volume"] # type: ignore

    # Convert from wide format to long format
    close_long = close.stack().rename("Close")
    volume_long = volume.stack().rename("Volume")

    batch_df = pd.concat([close_long, volume_long], axis=1).reset_index()

    # yfinance gives columns like Date, Ticker
    batch_df.columns = ["Date", "Symbol", "Close", "Volume"]

    all_data.append(batch_df)

final_df = pd.concat(all_data, ignore_index=True)

# Optional: sort output
final_df = final_df.sort_values(["Date", "Symbol"])

# Save to CSV
final_df.to_csv("sp500_10y_close_volume.csv", index=False)