import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from Model import ForexTransformer
from sklearn.model_selection import train_test_split
from twelvedata import TDClient 
from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Parameters
FILE_PATH = './Dataset/EURUSD/EURUSD_M30_features+label_v.2.1.csv'
API_KEY = 'e6b3ac6147a04e34bca8d5f1011b728c'
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = MinMaxScaler()
td = TDClient(apikey=API_KEY)
mint = 5

# Model definition
class ForexModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout):
        super(ForexModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        hn = self.dropout(hn[-1])
        out = self.fc(hn)
        return out
    
model = ForexModel(input_size, hidden_size, num_classes, dropout).to(device)

MODEL_PATH = "D:/Programing/AI Trader/Model/lstmModelv.1.0/Model v.1.0_loss 62.9028_Acc 0.8692_at 20241224-051604.model"

model.load_state_dict(torch.load(MODEL_PATH, weights_only=False))

model.eval()

print(f"{MODEL_PATH.split('/')[-1]} is loaded.")


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Mock function to simulate fetching real-time data
def fetch_realtime_data(interval="30min"):

    ts = td.time_series(symbol="EUR/USD", interval=interval, outputsize=seq_length).with_sma(time_period=200).with_sma(time_period=50).with_rsi(time_period=14).as_pandas()
    
    return ts

def predict(model, real_time_data, seq_length, scaler, device):
    real_time_data = scaler.fit_transform(real_time_data[['close', 'sma1', 'sma2', 'rsi']].values).round(4)
    real_time_data = torch.tensor(real_time_data[-seq_length:], dtype=torch.float32).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(real_time_data)
        _, predicted = torch.max(output, 1)
        return predicted.item()

def get_model_prediction(mint=30):
    real_time_data = fetch_realtime_data(f"{mint}min")
    prediction = predict(model, real_time_data, seq_length, scaler, device)
    return conditions_map[prediction], mint








