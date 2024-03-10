
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


def train_model(data):
        
    # data.set_index('Timestamp', inplace=True)

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

    X_train_scaled=scl.fit_transform(X_train)
    X_train= X_train.assign(Timestamp=X_train_scaled[:, 0])
    X_train= X_train.assign(Price=X_train_scaled[:, 1])
    X_train= X_train.assign(RSI=X_train_scaled[:, 2])


    X_test_scaled=scl.transform(X_test)

    X_test= X_test.assign(Timestamp=X_test_scaled[:, 0])
    X_test= X_test.assign(Price=X_test_scaled[:, 1])
    X_test= X_test.assign(RSI=X_test_scaled[:, 2])

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



    # Update these dimensions based on your dataset
    input_dim = X.shape[1]
    hidden_dim = 32
    num_layers = 2
    output_dim = 3  # Number of classes

    # Create the model
    model = LSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim)

    # Use CrossEntropyLoss for multi-class classification
    # Initialize the loss function with class weights
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Optimizer
    optimiser = torch.optim.Adam(model.parameters(), lr=0.05)

    # Assuming `optimizer` is your optimizer (e.g., Adam)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.8)



    num_epochs = 30


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

    return model, x_test



def save_model(model):
    # Define a path to save the model
    model_path = 'trained_model_lstm.pth'
    # Save the model's state dictionary
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
    file_path = 'Performance-test_2.csv'

    # Step 1: Read data
    data = read_data(file_path)

    # Step 2: Train model
    trained_model, x_test = train_model(data)

    # Step 3: Save model
    save_model(trained_model)
    
    backtest(trained_model, x_test)
