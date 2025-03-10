from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.stream import TradingStream
import matplotlib.pyplot as plt

# Import the Alpaca API key and secret key from a config file   
import config

client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)
account = dict(client.get_account())
for key, value in account.items():
    print(f"{key}: {value}")

# Place a market order to buy 100  shares of Apple
order_details = MarketOrderRequest(
    symbol="AAPL",
    qty=100,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY
)

# order = client.submit_order(order_data=order_details) 

# trades = TradingStream(config.API_KEY, config.SECRET_KEY, paper=True)
# async def trade_status(data):
#     print(data)

# trades.subscribe_trade_updates(trade_status)
# trades.run()

assets = [asset for asset in client.get_all_positions()]
positions = [(asset.symbol, asset.qty, asset.current_price) for asset in assets]
print("Positions")
print(f"{'Symbol':9} {'Qty':>4} {'Current Price':>15}")
print("-" * 28)
for position in positions:
    print(f"{position[0]:9} {float(position[1]) * float(position[2]):>15.2f}")

client.close_all_positions(cancel_orders=True)
print("Closed all positions")
# print("Cancelled all orders")
# orders = client.get_all_orders()  


