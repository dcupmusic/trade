
import numpy as np
import random
import pandas as pd 
from pylab import mpl, plt
import json
import joblib

import math, time
import itertools
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import joblib
from datetime import date
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import vectorbtpro as vbt
vbt.settings.set_theme('dark')
vbt.settings['plotting']['layout']['width'] = 1400
vbt.settings['plotting']['layout']['height'] = 700


import warnings
warnings.simplefilter("ignore", UserWarning)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

from functions import process_data, plot_target, tt_split

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
    


def read_data(file_path):
    return pd.read_csv(file_path)


def train_model(x_train, y_train, pos_weight):
    
    # Update these dimensions based on your dataset
    input_dim = x_train.shape[2]
    hidden_dim = 32
    num_layers = 2
    output_dim = 1  # Number of classes
    dropout_rate=0.05
    
    num_epochs = 100
    
    
    model = BiLSTMClassifierWithAttention(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim, dropout_rate=dropout_rate)

    # class_weights_tensor = torch.tensor([1, 1], dtype=torch.float)

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimiser = torch.optim.Adam(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=50, gamma=0.9)

    model.train()
    for t in range(num_epochs):
        y_train_pred = model(x_train)
        
        loss = loss_fn(y_train_pred.squeeze(), y_train.float())
        
        if t % 20 == 0:
            print("Epoch ", t, "Loss: ", loss.item())
            
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        scheduler.step()

    return model



def save_model(model):
    # Define a path to save the model
    model_path = 'trained_model_lstm_binary.pth'
    # Save the model's state dictionary
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

def backtest(df, model, x_test ,indices_test):
    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test)

        probabilities = torch.sigmoid(y_test_pred).squeeze()
        predicted_labels = (probabilities > 0.5).long()
        predicted_labels_numpy = predicted_labels.cpu().numpy()


    df_split = df[-len(predicted_labels_numpy):].copy()
    df_split.loc[:, "signal"] = predicted_labels_numpy
    signal = df_split['signal']
    entries = signal == 1
    exits = signal == 0
    pf = vbt.Portfolio.from_signals(
        close=df_split.price, 
        long_entries=entries, 
        long_exits=exits,
        size=100,
        size_type='value',
        init_cash='auto', 
    )

    print(pf.stats()["Total Return [%]"])
    pf.plot({"orders", "cum_returns", }, settings=dict(bm_returns=False)).show()

# Main script
if __name__ == "__main__":
    # Replace 'your_data.csv' with the actual file path
    file_path = '../data/1ySOLdata1hAllHassInd.csv'

    coin = 'SOL'
    # Step 1: Read data
    data = read_data(file_path)
    
    df = process_data(data, 30, 'SOL')


    
    window_size = 10
    timestep = 20
    
    # plot_target(df)
    
    x_train, x_test, y_train, y_test, pos_weight = tt_split(df, timestep, coin)
    


    # # Step 2: Train model
    trained_model = train_model(x_train, y_train, pos_weight)

    # # Step 3: Save model
    save_model(trained_model)
    
    # backtest(df, trained_model, x_test, indices_test)
