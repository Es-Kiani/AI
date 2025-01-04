from Model2 import *

# Parameters
FILE_PATH = './Dataset/EURUSD/EURUSD_M30_features+label_v.2.1.csv'
API_KEY = 'e6b3ac6147a04e34bca8d5f1011b728c'
PAIR = "EUR/USD"
COLUMNS = ['Close', 'SMA200', 'SMA50', 'RSI14']
LABEL = 'signal'
seq_length = 20
batch_size = 1024
epochs = 100
dropout = 0.4
learning_rate = 0.01

input_size = 4  # ['Close', 'SMA200', 'SMA50', 'RSI14']
hidden_size = 128
num_classes = 3  # [buy, sell, nothing]
conditions_map = {0: "Buy", 1: "Sell", 2: "Nothing"}

account_number = 20037567
password = '1380_EskI'
server = "WMMarkets-Demo"
VOTE_LEN = 3
OPEN_POSITIONS = 4
RR_MIN_RATIO = 2    # Not in Use
TP_PERCENT = 0.0618
SL_PERCENT = 0.0300
symbol = PAIR.split('/')[0] + PAIR.split('/')[1]
volume = 0.25

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = MinMaxScaler()
td = TDClient(apikey=API_KEY)
time_frame = 15