
import numpy as np
import random
import pandas as pd 
from pylab import mpl, plt


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

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import vectorbtpro as vbt
vbt.settings.set_theme('dark')
vbt.settings['plotting']['layout']['width'] = 800
vbt.settings['plotting']['layout']['height'] = 400

from functions import process_data, plot_target, tt_split, save_model, read_data


import warnings
warnings.simplefilter("ignore", UserWarning)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        
        return out
    
    



def train_model(x_train, y_train, class_weights_tensor):
    
    # Update these dimensions based on your dataset
    input_dim = x_train.shape[2]
    hidden_dim = 32
    num_layers = 2
    output_dim = 3  # Number of classes

    model = LSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimiser = torch.optim.Adam(model.parameters(), lr=0.03)

    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=50, gamma=0.9)



    num_epochs = 100


    model.train()
    for t in range(1, num_epochs+1):

        y_train_pred = model(x_train)
        loss = loss_fn(y_train_pred, y_train.long()) 
        
        if t % 50 == 0:
            print("Epoch ", t, "Loss: ", loss.item())
            save_model(model, t)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        scheduler.step()    

    return model





# Main script
if __name__ == "__main__":
    # Replace 'your_data.csv' with the actual file path
    file_path = '../data/1ySOLdata1hAllHassInd.csv'

    # Step 1: Read data
    data = read_data(file_path)
    
    best_features = ['timestamp', 'price', 'Close', 'Volume', 'tema_12', 'abands_upper', 'abands_middle', 'adxr', 'ao', 'apo', 'aroonosc', 'atr', 'avgprice', 'bbands_upper', 'bbands_lower', 'bop', 'cci', 'cmo', 'dema', 'dx', 'fastrsi', 'ht_trendline', 'kama', 'keltner_lower', 'kri', 'macdfix_hist', 'minusdi', 'stochf_fastD', 'stochrsi_fastK', 'stochrsi_fastD', 'trange', 'trix', 'tsi', 'udrsi', 'willr']
    dfcopy = data[best_features].copy()
    
    window_size = 50
    timestep = 20
    df = process_data(dfcopy, window_size)
    
    # vbdata = vbt.Data.from_data(df)
    # plot_target(vbdata.data['symbol'])
    short_weight = 1.05
    none_weight = 0
    long_weight = 1
    
    x_train, x_test, y_train, y_test, class_weights_tensor = tt_split(df, short_weight, none_weight, long_weight, timestep)


    
    trained_model = train_model(x_train, y_train, class_weights_tensor)

    # save_model(trained_model, 95)
