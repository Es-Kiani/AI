from Model3 import *
from Model3_config import *


# Mock function to simulate fetching real-time data
def fetch_realtime_data(interval="30min", pair=PAIR):

    ts = td.time_series(symbol=pair, interval=interval, outputsize=seq_length).with_sma(time_period=200).with_sma(time_period=50).with_rsi(time_period=14).as_pandas()
    
    return ts

def predict(model, real_time_data, seq_length, scaler, device):
    real_time_data = scaler.fit_transform(real_time_data[['close', 'sma1', 'sma2', 'rsi']].values)
    real_time_data = torch.tensor(real_time_data[-seq_length:], dtype=torch.float32).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(real_time_data)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# real_time_data = fetch_realtime_data(f"{time_frame}min")
# real_time_data[['close', 'sma1', 'sma2', 'rsi']].values

# while True:
#     try:
#         real_time_df = fetch_realtime_data(f"{time_frame}min")
#         predicted_signal = predict(model, real_time_df, seq_length, scaler, device)
#         prediction = conditions_map[predicted_signal]
        
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         print(f"Real-time Prediction -> {prediction}\t|\t{timestamp}")
        
#     except ValueError as e:
#         print(e)
    
#     time.sleep(time_frame*60)

def connect_to_metatrader():
    if not mt5.initialize():
        print("Failed to initialize MetaTrader5")
        quit()
        return

def get_open_positions():
    return mt5.positions_total()

def calculate_tp_sl(price, is_buy, tp_percentage=TP_PERCENT, sl_percentage=SL_PERCENT):
    if is_buy:
        tp = price * (1 + tp_percentage / 100)
        sl = price * (1 - sl_percentage / 100)
    else:
        tp = price * (1 - tp_percentage / 100)
        sl = price * (1 + sl_percentage / 100)
    return tp, sl

def place_trade(symbol, action, volume, tp, sl):
    if action == "buy": 
        order_type = mt5.ORDER_TYPE_BUY
    elif action == "sell":
        order_type = mt5.ORDER_TYPE_SELL
    else:
        print("Not Execute Any Order!")
    
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": mt5.symbol_info_tick(symbol).ask if action == "buy" else mt5.symbol_info_tick(symbol).bid,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 123456,  # Unique ID for the EA or script
        "comment": "Majority vote trade",
    }
    result = mt5.order_send(request)
    return result

def open_position(vote):
    connect_to_metatrader()
    vote = vote.lower()
    
    if vote not in ["buy", "sell"]:
        print("No clear majority vote. No trade executed.")
        return

    if get_open_positions() >= OPEN_POSITIONS:
        print("Maximum position limit reached. No trade executed.")
        return

    price = mt5.symbol_info_tick(symbol).ask if vote == "buy" else mt5.symbol_info_tick(symbol).bid
    tp, sl = calculate_tp_sl(price, is_buy=(vote == "buy"))

    result = place_trade(symbol, vote, volume, tp, sl)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"Trade executed successfully: {vote} {volume} lots of {symbol}")
    else:
        print(f"Trade failed. Error code: {result.retcode}")

    mt5.shutdown()





connect_to_metatrader()

if not mt5.login(account_number, password, server):
    print(f"Failed to log in to account {account_number}")
    print(f"Error code: {mt5.last_error()}")
    mt5.shutdown()
