
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



class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer to output a single value
        self.fc = nn.Linear(hidden_dim, 1)  # Output dim is 1 for binary classification
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        
        # No sigmoid activation layer here, nn.BCEWithLogitsLoss will be applied outside the model
        
        return out
    
    

def read_data(file_path):
    return pd.read_csv(file_path)


def process_data(data):
        

    data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')
    data.set_index('Timestamp', inplace=True)
    pd.set_option('future.no_silent_downcasting', True)
    
    window_size = 20
    data_trimmed = data.drop('Signal', axis=1)
    data_trimmed.loc[:, 'Signal'] = 'SignalNone'

    rolling_max = data_trimmed.loc[:,'Price'].rolling(window=2*window_size+1, center=True, min_periods=1).max()
    rolling_min = data_trimmed.loc[:,'Price'].rolling(window=2*window_size+1, center=True, min_periods=1).min()

    is_peak = (data_trimmed.loc[:, 'Price'] == rolling_max)

    is_low = (data_trimmed.loc[:, 'Price'] == rolling_min) 

    # Update signal columns where conditions are met
    data_trimmed.loc[is_peak, 'Signal'] = 'SignalShort'
    data_trimmed.loc[is_low, 'Signal'] = 'SignalLong'
    df = data_trimmed.copy()

    df_filtered = df[df['Signal'] != 'SignalNone']

    # Iterate through the DataFrame and adjust the signals
    for i in range(1, len(df_filtered)):
        current_signal = df_filtered.iloc[i]['Signal']
        previous_signal = df_filtered.iloc[i - 1]['Signal']
        current_close = df_filtered.iloc[i]['Price']
        previous_close = df_filtered.iloc[i - 1]['Price']
        
        if current_signal == previous_signal:
            if current_signal == 'SignalLong' and previous_close > current_close:
                df_filtered.iloc[i - 1, df_filtered.columns.get_loc('Signal')] = 'SignalNone'
            elif current_signal != 'SignalLong' and previous_close < current_close:
                df_filtered.iloc[i - 1, df_filtered.columns.get_loc('Signal')] = 'SignalNone'
            else:
                df_filtered.iloc[i, df_filtered.columns.get_loc('Signal')] = 'SignalNone'


    df.update(df_filtered)

    previous_signal = None

    for i in range(len(df)):
        if df.iloc[i, df_filtered.columns.get_loc('Signal')] == "SignalNone" and previous_signal is not None:
            df.iloc[i, df_filtered.columns.get_loc('Signal')] = previous_signal 
        elif df.iloc[i, df_filtered.columns.get_loc('Signal')] != "SignalNone":
            previous_signal = df.iloc[i, df_filtered.columns.get_loc('Signal')]

    df = df.loc[df['Signal'] != 'SignalNone']

    df['Signal'] = df['Signal'].replace({'SignalLong': 1, 'SignalShort': 0})
    df = df.ffill()


    return df

def tt_split(df):

    X = df.iloc[:, :-1]

    y = df.iloc[:, -1]

    indices = np.arange(X.shape[0])

    test_size = int(0.3*(len(X)))
    X_train = X[:-test_size]
    X_test = X[-test_size:]

    y_train = y[:-test_size]
    y_test = y[-test_size:]
    
    indices_train = indices[:-test_size]
    indices_test = indices[-test_size:]


    scl = StandardScaler()
    
    
    features = X_train.columns

    X_train_scaled = scl.fit_transform(X_train)
    X_test_scaled = scl.transform(X_test)


    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)
    X_train = X_train_scaled_df
    X_test = X_test_scaled_df


    timestep = 20
    X_train_list = []
    y_train_list = []

    
    for i in range(timestep, len(X_train) - timestep + 1):
        X_train_list.append(np.array(X_train.iloc[i-timestep:i]))
        y_train_list.append(y_train.iloc[i])

    X_test_list = []
    y_test_list = []

    for i in range(timestep, len(X_test) - timestep + 1):
        X_test_list.append(np.array(X_test.iloc[i-timestep:i]))
        y_test_list.append(y_test.iloc[i])


    x_train = np.array(X_train_list)
    x_test = np.array(X_test_list)  

    y_train = np.array(y_train_list)
    y_test = np.array(y_test_list)

    indices_train_adjusted = indices_train[timestep:len(indices_train) - timestep + 1]
    indices_test_adjusted = indices_test[timestep:len(indices_test) - timestep + 1]

    # make training and test sets in torch
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    # saving param scales
    scaler_params = {'mean': scl.mean_, 'scale': scl.scale_}
    joblib.dump(scaler_params, 'scaler_params_binary.joblib')

    return x_train, x_test, y_train, y_test, indices_train_adjusted, indices_test_adjusted


def train_model(x_train, y_train):
    
    # Update these dimensions based on your dataset
    input_dim = x_train.shape[2]
    hidden_dim = 32
    num_layers = 2
    output_dim = 1  # Number of classes

    num_epochs = 100
    
    
    model = LSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)

    # class_weights_tensor = torch.tensor([1, 1], dtype=torch.float)

    loss_fn = torch.nn.BCEWithLogitsLoss()

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
    df_split.loc[:, "Signal"] = predicted_labels_numpy
    signal = df_split['Signal']
    entries = signal == 1
    exits = signal == 0
    pf = vbt.Portfolio.from_signals(
        close=df_split.Price, 
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
    file_path = '../data/Performance-test_2.csv'

    # Step 1: Read data
    data = read_data(file_path)
    
    df = process_data(data)

    x_train, x_test, y_train, y_test, indices_train, indices_test = tt_split(df)

    # Step 2: Train model
    trained_model = train_model(x_train, y_train)

    # Step 3: Save model
    save_model(trained_model)
    
    backtest(df, trained_model, x_test, indices_test)
