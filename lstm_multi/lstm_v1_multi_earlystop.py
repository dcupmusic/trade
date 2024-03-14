
import numpy as np
import random
import pandas as pd 
from pylab import mpl, plt



import json
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
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
        

    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    data.set_index('timestamp', inplace=True)
    pd.set_option('future.no_silent_downcasting', True)
    
    window_size = 8
    data_trimmed = data.copy()
    data_trimmed.loc[:, 'Signal'] = 'SignalNone'

    rolling_max = data_trimmed.loc[:,'price'].rolling(window=2*window_size+1, center=True, min_periods=1).max()
    rolling_min = data_trimmed.loc[:,'price'].rolling(window=2*window_size+1, center=True, min_periods=1).min()

    is_peak = (data_trimmed.loc[:, 'price'] == rolling_max)

    is_low = (data_trimmed.loc[:, 'price'] == rolling_min) 

    # Update signal columns where conditions are met
    data_trimmed.loc[is_peak, 'Signal'] = 'SignalShort'
    data_trimmed.loc[is_low, 'Signal'] = 'SignalLong'
    df = data_trimmed.copy()

    df_filtered = df[df['Signal'] != 'SignalNone']

    # Iterate through the DataFrame and adjust the signals
    for i in range(1, len(df_filtered)):
        current_signal = df_filtered.iloc[i]['Signal']
        previous_signal = df_filtered.iloc[i - 1]['Signal']
        current_close = df_filtered.iloc[i]['price']
        previous_close = df_filtered.iloc[i - 1]['price']
        
        if current_signal == previous_signal:
            if current_signal == 'SignalLong' and previous_close > current_close:
                df_filtered.iloc[i - 1, df_filtered.columns.get_loc('Signal')] = 'SignalNone'
            elif current_signal != 'SignalLong' and previous_close < current_close:
                df_filtered.iloc[i - 1, df_filtered.columns.get_loc('Signal')] = 'SignalNone'
            else:
                df_filtered.iloc[i, df_filtered.columns.get_loc('Signal')] = 'SignalNone'


    df.update(df_filtered)


    df['Signal'] = df['Signal'].replace({'SignalShort': 0, 'SignalNone': 1, 'SignalLong': 2})
    df = df.ffill()

    feature_names = [col for col in df.columns if col != 'Signal']

    # Save feature names to a JSON file
    with open('feature_names.json', 'w') as f:
        json.dump(feature_names, f)

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
        
    unique_labels = np.unique(y_train_np)
    class_weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=y_train_np)

    index_of_class_short = np.where(unique_labels == 0)[0][0]
    index_of_class_none = np.where(unique_labels == 1)[0][0]
    index_of_class_long = np.where(unique_labels == 2)[0][0]

    class_weights[index_of_class_short] *= 1
    class_weights[index_of_class_none] *= 0.3
    class_weights[index_of_class_long] *= 1

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    class_weights_tensor = class_weights_tensor.to('cpu')

    
    return x_train, x_test, y_train, y_test, class_weights_tensor

def save_model(model, epoch):
    model_path = f'trained_model_lstm_epoch_{epoch}.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')
    
    
def train_model(x_train, y_train, class_weights_tensor):
    
    input_dim = x_train.shape[2]
    hidden_dim = 32
    num_layers = 2
    output_dim = 3
    dropout_rate = 0.1


    model = BiLSTMClassifierWithAttention(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim, dropout_rate=dropout_rate)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimiser = torch.optim.Adam(model.parameters(), lr=0.02)

    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=5, gamma=0.9)


    num_epochs = 95
    
    model.train()
    for t in range(num_epochs):

        y_train_pred = model(x_train)
        loss = loss_fn(y_train_pred, y_train.long())  
        
        if (t % 5 == 0) and (t >= 9):
            print("Epoch ", t, "Loss: ", loss.item())
            # save_model(model, t)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        scheduler.step()    

    return model



def backtest(df, model, x_test):
    model.eval()
    with torch.no_grad():
        
        y_test_pred = model(x_test)
        probabilities = torch.softmax(y_test_pred, dim=1)
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
    stats = pf.stats()
    
    total_return = stats['Total Return [%]']
    total_return = 0 if np.isnan(total_return) else total_return
    # pf.plot({"orders", "cum_returns", }, settings=dict(bm_returns=False)).show()
    model.train()
    return total_return

# Main script
if __name__ == "__main__":
    # Replace 'your_data.csv' with the actual file path
    file_path = '../data/1ySOLdata1hAllHassInd.csv'

    # Step 1: Read data
    data = read_data(file_path)
    
    df = process_data(data)
    
    x_train, x_test, y_train, y_test, class_weights_tensor = tt_split(df)

    trained_model = train_model(x_train, y_train, class_weights_tensor)

    save_model(trained_model, 95)

    
    
