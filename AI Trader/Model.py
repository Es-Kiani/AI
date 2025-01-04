import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.cluster import KMeans
from datetime import datetime
from tqdm import tqdm


# Positional Encoding class for the Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

class ForexTransformer(nn.Module):
    def __init__(self, input_dim, seq_length, num_heads, num_layers, output_dim):
        super(ForexTransformer, self).__init__()
        self.embed_dim = 256  # Increased embedding dimension
        self.embedding = nn.Linear(input_dim, self.embed_dim)
        self.pos_encoder = PositionalEncoding(self.embed_dim, max_len=seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(0.4)  # Increased dropout rate
        self.fc = nn.Linear(self.embed_dim * seq_length, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.dropout(x)  # Apply dropout after transformer
        x = x.flatten(start_dim=1)  # Flatten for the fully connected layer
        x = self.fc(x)
        return x
