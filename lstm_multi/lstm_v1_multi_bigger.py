
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


import warnings
warnings.simplefilter("ignore", UserWarning)

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
    
    

def read_data(file_path):
    return pd.read_csv(file_path)


def process_data(data):
        

    data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')
    data.set_index('Timestamp', inplace=True)
    pd.set_option('future.no_silent_downcasting', True)
    
    window_size = 15
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


    df['Signal'] = df['Signal'].replace({'SignalNone': 1, 'SignalLong': 2, 'SignalShort': 0})
    df = df.ffill()


    return df


def tt_split(df):

    X = df.iloc[:, :-1]

    y = df.iloc[:, -1]

    test_size = int(0.3*(len(X)))
    X_train = X[:-test_size]
    X_test = X[-test_size:]

    y_train = y[:-test_size]
    y_test = y[-test_size:]


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


    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()

    scaler_params = {'mean': scl.mean_, 'scale': scl.scale_}
    joblib.dump(scaler_params, 'scaler_params.joblib')
 

    if isinstance(y_train, torch.Tensor):
        y_train_np = y_train.cpu().numpy()
    else:
        y_train_np = y_train

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_np), y=y_train_np)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)


    class_weights_tensor = class_weights_tensor.to('cpu')
    
    return x_train, x_test, y_train, y_test, class_weights_tensor


def train_model(x_train, y_train, class_weights_tensor):
    
    input_dim = x_train.shape[2]
    hidden_dim = 32
    num_layers = 2
    output_dim = 3
    dropout_rate = 0.1

    # Create the model
    model = BiLSTMClassifierWithAttention(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim, dropout_rate=dropout_rate)

    # Use CrossEntropyLoss for multi-class classification
    # Initialize the loss function with class weights
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Optimizer
    optimiser = torch.optim.Adam(model.parameters(), lr=0.03)

    # Assuming `optimizer` is your optimizer (e.g., Adam)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=50, gamma=0.9)



    num_epochs = 60


    model.train()
    for t in range(num_epochs):
        # Forward pass: Compute predicted y by passing x to the model
        y_train_pred = model(x_train)

        # Compute loss
        loss = loss_fn(y_train_pred, y_train.long())  # Ensure y_train is of type torch.long
        if t % 20 == 0:
            print("Epoch ", t, "Loss: ", loss.item())

        # Zero gradients before backward pass
        optimiser.zero_grad()

        # Perform backward pass: compute gradients of the loss with respect to all the learnable parameters
        loss.backward()

        # Update the parameters using the gradients and optimizer algorithm
        optimiser.step()
        
        # Step the scheduler
        scheduler.step()    

    return model



def save_model(model):
    # Define a path to save the model
    model_path = 'trained_model_lstm.pth'
    # Save the model's state dictionary
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

def backtest(df, model, x_test):
    model.eval()
    with torch.no_grad():
        
        y_test_pred = model(x_test)
        # Convert logits to probabilities
        probabilities = torch.softmax(y_test_pred, dim=1)

        # Get the predicted class labels
        _, predicted_labels = torch.max(probabilities, 1)
    predicted_labels_numpy = predicted_labels.numpy()


    df_split = df[-len(predicted_labels_numpy):].copy()
    df_split.loc[:, "Signal"] = predicted_labels_numpy
    signal = df_split['Signal']
    entries = signal == 2
    exits = signal == 0
    pf = vbt.Portfolio.from_signals(
        close=df_split.Price, 
        long_entries=entries, 
        long_exits=exits,
        size=100,
        size_type='value',
        init_cash='auto'
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
    
    x_train, x_test, y_train, y_test, class_weights_tensor = tt_split(df)

    # Step 2: Train model
    trained_model = train_model(x_train, y_train, class_weights_tensor)

    # Step 3: Save model
    save_model(trained_model)
    
    backtest(df, trained_model, x_test)
