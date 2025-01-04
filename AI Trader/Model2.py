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
from Model2_config import *


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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)