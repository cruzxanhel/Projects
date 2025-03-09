import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import config

# Initialize Alpaca Trading Client
client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)

# Initialize Alpaca Data Client (for fetching historical stock data)
data_client = StockHistoricalDataClient(config.API_KEY, config.SECRET_KEY)

# Function to fetch historical stock data
def get_historical_data(symbol, days=100):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Request historical bars
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date
    )
    
    bars = data_client.get_stock_bars(request_params).df

    if symbol in bars.index:
        df = bars.loc[symbol].reset_index()
        df.rename(columns={"timestamp": "date", "close": "c", "high": "h", "low": "l", "volume": "v"}, inplace=True)
        return df
    else:
        print(f"Error: No data for {symbol}")
        return None

# Function to calculate indicators
def calculate_indicators(df):
    df["SMA_10"] = df["c"].rolling(window=10, min_periods=1).mean()
    df["SMA_50"] = df["c"].rolling(window=50, min_periods=1).mean()

    # RSI Calculation
    delta = df["c"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=1).mean()
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD Calculation
    short_ema = df["c"].ewm(span=12, adjust=False).mean()
    long_ema = df["c"].ewm(span=26, adjust=False).mean()
    df["MACD"] = short_ema - long_ema
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df

# Swing Trading Strategy
def swing_trade(symbol):
    df = get_historical_data(symbol)
    if df is None:
        return

    df = calculate_indicators(df)
    print(f"\n--- {symbol} Latest Indicators ---")
    print(df.tail(3)[["date", "c", "SMA_10", "SMA_50", "RSI", "MACD", "Signal"]])  

    # Buy conditions
    buy_condition = (
    df["SMA_10"].iloc[-1] > df["SMA_50"].iloc[-1] or 
    df["RSI"].iloc[-1] > 35  
)

    # Sell conditions
    sell_condition = (
    df["SMA_10"].iloc[-1] < df["SMA_50"].iloc[-1] or 
    df["RSI"].iloc[-1] < 65  
)
    print(f"\nChecking {symbol} for trade opportunities...")
    
    # Execute buy order
    if buy_condition:
        print(f"Buying {symbol} - SMA: {df['SMA_10'].iloc[-1]:.2f} > {df['SMA_50'].iloc[-1]:.2f}, RSI: {df['RSI'].iloc[-1]:.2f}, MACD: {df['MACD'].iloc[-1]:.2f}")
        order = MarketOrderRequest(symbol=symbol, qty=10, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
        client.submit_order(order_data=order)

    # Execute sell order
    elif sell_condition:
        print(f"Selling {symbol} - SMA: {df['SMA_10'].iloc[-1]:.2f} < {df['SMA_50'].iloc[-1]:.2f}, RSI: {df['RSI'].iloc[-1]:.2f}, MACD: {df['MACD'].iloc[-1]:.2f}")
        order = MarketOrderRequest(symbol=symbol, qty=10, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
        client.submit_order(order_data=order)
    else:
        print(f"No trade opportunity for {symbol}.")
    
# Run strategy for a list of stocks
stocks = ["AAPL", "MSFT", "NVDA", "RGTI", "TSLA", "AMZN", "GOOGL", "META", "NFLX", "BABA", "DIS", "V", "JPM", "WMT", "PG"]
for stock in stocks:
    swing_trade(stock)
