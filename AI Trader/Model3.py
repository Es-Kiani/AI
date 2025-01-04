import torch.nn as nn
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
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
from collections import deque
import MetaTrader5 as mt5
from Model3_config import *


class CNNLSTM(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch, features, sequence)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # Change shape to (batch, sequence, features)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last output of LSTM
        x = self.fc(x)
        return x



model = CNNLSTM(num_features=input_size, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)