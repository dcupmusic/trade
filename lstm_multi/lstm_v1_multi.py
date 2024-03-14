
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
    
    

def read_data(file_path):
    return pd.read_csv(file_path)


def process_data(data):
        

    data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')
    data.set_index('Timestamp', inplace=True)
    pd.set_option('future.no_silent_downcasting', True)

    data['Signal'] = data['Signal'].replace({'SignalNone': 1, 'SignalLong': 2, 'SignalShort': 0})
    data.ffill()

    data = data.copy()

    X = data.iloc[:, :-1]

    y = data.iloc[:, -1]

    test_size = int(0.3*(len(X)))
    X_train = X[:-test_size]
    X_test = X[-test_size:]

    y_train = y[:-test_size]
    y_test = y[-test_size:]


    scl = StandardScaler()
    
    
    features = X_train.columns
    # Fit the scaler on the training data and transform it
    X_train_scaled = scl.fit_transform(X_train)
    X_test_scaled = scl.transform(X_test)

    # Convert the scaled arrays back into DataFrames with the original column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)
    X_train = X_train_scaled_df
    X_test = X_test_scaled_df


    timestep = 20
    X_train_list = []
    y_train_list = []

    # Adjust the range to stop at the last point where a full timestep can be created
    for i in range(timestep, len(X_train) - timestep + 1):  # Adjust the loop to stop earlier
        X_train_list.append(np.array(X_train.iloc[i-timestep:i]))
        # Only append the next value instead of a range of values
        y_train_list.append(y_train.iloc[i])  # Assuming you want the next value as the target

    X_test_list = []
    y_test_list = []

    for i in range(timestep, len(X_test) - timestep + 1):  # Adjust the loop to stop earlier
        X_test_list.append(np.array(X_test.iloc[i-timestep:i]))
        # Only append the next value instead of a range of values
        y_test_list.append(y_test.iloc[i])  # Assuming you want the next value as the target



    x_train = np.array(X_train_list)
    x_test = np.array(X_test_list)  

    y_train = np.array(y_train_list)
    y_test = np.array(y_test_list)


    # make training and test sets in torch
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()

    # saving param scales
    scaler_params = {'mean': scl.mean_, 'scale': scl.scale_}
    joblib.dump(scaler_params, 'scaler_params.joblib')
 

    # Convert y_train to a numpy array if it's a tensor
    if isinstance(y_train, torch.Tensor):
        y_train_np = y_train.cpu().numpy()
    else:
        y_train_np = y_train  # Assuming y_train is already a numpy array

    # Calculate class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_np), y=y_train_np)

    # Convert class weights to a tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)


    # Move class weights to the same device as your model and data
    class_weights_tensor = class_weights_tensor.to('cpu')  # device could be 'cpu' or 'cuda'
    
    return x_train, x_test, y_train, y_test, class_weights_tensor


def train_model(x_train, y_train, class_weights_tensor):
    
    # Update these dimensions based on your dataset
    input_dim = x_train.shape[2]
    hidden_dim = 32
    num_layers = 2
    output_dim = 3  # Number of classes

    # Create the model
    model = LSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim)

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

        y_train_pred = model(x_train)

        # Compute loss
        loss = loss_fn(y_train_pred, y_train.long()) 
        if t % 5 == 0:
            print("Epoch ", t, "Loss: ", loss.item())
            save_model(model, t)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        scheduler.step()    

    return model



def save_model(model, epoch):
    model_path = f'trained_model_lstm_epoch_{epoch}.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

def backtest(model, x_test):
    model.eval()
    with torch.no_grad():
        
        y_test_pred = model(x_test)
        # Convert logits to probabilities
        probabilities = torch.softmax(y_test_pred, dim=1)

        # Get the predicted class labels
        _, predicted_labels = torch.max(probabilities, 1)
    predicted_labels_numpy = predicted_labels.numpy()


    df_split = data[-len(predicted_labels_numpy):].copy()
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
    
    x_train, x_test, y_train, y_test, class_weights_tensor = process_data(data)

    # Step 2: Train model
    trained_model = train_model(x_train, y_train, class_weights_tensor)

    # Step 3: Save model
    # save_model(trained_model)
    
    # backtest(trained_model, x_test)
