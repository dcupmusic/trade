import numpy as np
import random
import pandas as pd 
from pylab import mpl, plt

import json

from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import joblib
from datetime import date

import vectorbtpro as vbt
vbt.settings.set_theme('dark')
vbt.settings['plotting']['layout']['width'] = 800
vbt.settings['plotting']['layout']['height'] = 400

import warnings
warnings.simplefilter("ignore", UserWarning)

from functions import process_data, plot_target, tt_split, save_model, read_data


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output shape: [batch_size, seq_length, hidden_dim]
        weights = torch.tanh(self.linear(lstm_output))
        weights = F.softmax(weights, dim=1)
        
        # Context vector with weighted sum
        context = weights * lstm_output
        context = torch.sum(context, dim=1)
        return context, weights

class BiLSTMClassifierWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate):
        super(BiLSTMClassifierWithAttention, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        
        # Batch Normalization Layer for Conv1d
        self.bn_conv1 = nn.BatchNorm1d(hidden_dim)
        
        # LSTM Layer
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        
        # Attention Layer
        self.attention = Attention(hidden_dim * 2)  # For bidirectional LSTM
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # Adjusted for attention context vector
        
        # Batch Normalization Layer for FC1
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Output layer
        
        # Additional Dropout for the fully connected layer
        self.dropout_fc = nn.Dropout(dropout_rate / 2)

    def forward(self, x):
        # Reshape x for Conv1d
        x = x.permute(0, 2, 1)
        
        # Convolutional layer
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.relu(x)
        
        # Reshape back for LSTM
        x = x.permute(0, 2, 1)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        
        # Applying attention mechanism to LSTM outputs
        context, _ = self.attention(out)
        
        # Fully connected layers using the context vector from attention
        out = self.fc1(context)
        out = self.bn_fc1(out)
        out = F.relu(out)
        out = self.dropout_fc(out)
        out = self.fc2(out)
        
        return out
    
    
   
def train_model(x_train, y_train, class_weights_tensor, num_epochs):
    
    input_dim = x_train.shape[2]
    hidden_dim = 32
    num_layers = 2
    output_dim = 3
    dropout_rate = 0.1


    model = BiLSTMClassifierWithAttention(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim, dropout_rate=dropout_rate)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimiser = torch.optim.Adam(model.parameters(), lr=0.02)

    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=50, gamma=0.9)

    
    model.train()
    for t in range(1, num_epochs+1):

        y_train_pred = model(x_train)
        loss = loss_fn(y_train_pred, y_train.long())  
        
        if t % num_epochs == 0:
            print("Epoch ", t, "Loss: ", loss.item())
            save_model(model, t, coin)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        scheduler.step()    

    return model


# Main script
if __name__ == "__main__":
    # Replace 'your_data.csv' with the actual file path
    coin = 'NEAR'
    
    file_path = f'../data/1y{coin}data1hAllHassInd.csv'

    # Step 1: Read data
    data = read_data(file_path)
    # best_features = ['timestamp', 'price', 'Close', 'Volume', 'tema_12', 'abands_upper', 'abands_middle', 'adxr', 'ao', 'apo', 'aroonosc', 'atr', 'avgprice', 'bbands_upper', 'bbands_lower', 'bop', 'cci', 'cmo', 'dema', 'dx', 'fastrsi', 'ht_trendline', 'kama', 'keltner_lower', 'kri', 'macdfix_hist', 'minusdi', 'stochf_fastD', 'stochrsi_fastK', 'stochrsi_fastD', 'trange', 'trix', 'tsi', 'udrsi', 'willr']
    best_features = ['timestamp', 'price', 'Close', 'Volume', 'adosc', 'macdfix_macd', 'keltner_middle', 'kri', 'bbands_middle', 'tema', 'ema', 'natr', 'bbands_upper', 'ad', 'bbands_lower', 'abands_middle', 'ht_dcperiod', 'atr', 'fastrsi', 'tema_12', 'tsi', 'dema', 'cmo', 'apo']

    dfcopy = data[best_features].copy()
    # dfcopy = data.copy()
    
    window_size = 30
    timestep = 40
    short_weight = 1.2
    none_weight = 0
    long_weight = 1

    
    df = process_data(dfcopy, window_size, coin)
    
    x_train, x_test, y_train, y_test, class_weights_tensor = tt_split(df, short_weight, none_weight, long_weight, timestep, coin)

    epochs = 100
    trained_model = train_model(x_train, y_train, class_weights_tensor, epochs)

    # vbdata = vbt.Data.from_data(df)
    # plot_target(vbdata.data['symbol'])
    # save_model(trained_model, 95)

    
    
