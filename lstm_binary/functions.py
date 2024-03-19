import pandas as pd
import numpy as np
import json
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


import vectorbtpro as vbt
vbt.settings.set_theme('dark')
vbt.settings['plotting']['layout']['width'] = 800
vbt.settings['plotting']['layout']['height'] = 400


def read_data(file_path):
    return pd.read_csv(file_path)




def process_data(data, window_size, coin):


    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    data.set_index('timestamp', inplace=True)
    pd.set_option('future.no_silent_downcasting', True)

    window_size = window_size
    data_trimmed = data.copy()
    data_trimmed.loc[:, 'signal'] = 'SignalNone'

    rolling_max = data_trimmed.loc[:,'price'].rolling(window=2*window_size+1, center=True, min_periods=1).max()
    rolling_min = data_trimmed.loc[:,'price'].rolling(window=2*window_size+1, center=True, min_periods=1).min()

    is_peak = (data_trimmed.loc[:, 'price'] == rolling_max)

    is_low = (data_trimmed.loc[:, 'price'] == rolling_min)

    data_trimmed.loc[is_peak, 'signal'] = 'SignalShort'
    data_trimmed.loc[is_low, 'signal'] = 'SignalLong'
    df = data_trimmed.copy()

    def filter_pivots(data):
      df_filtered = df[df['signal'] != 'SignalNone']


      for i in range(1, len(df_filtered)):
          current_signal = df_filtered.iloc[i]['signal']
          previous_signal = df_filtered.iloc[i - 1]['signal']
          current_close = df_filtered.iloc[i]['price']
          previous_close = df_filtered.iloc[i - 1]['price']

          if current_signal == previous_signal:
              if current_signal == 'SignalLong':
                  if previous_close > current_close:
                      df_filtered.iloc[i - 1, df_filtered.columns.get_loc('signal')] = 'SignalNone'
                  else:
                      df_filtered.iloc[i, df_filtered.columns.get_loc('signal')] = 'SignalNone'
              elif current_signal == 'SignalShort':
                  if previous_close < current_close:
                      df_filtered.iloc[i - 1, df_filtered.columns.get_loc('signal')] = 'SignalNone'
                  else:
                      df_filtered.iloc[i, df_filtered.columns.get_loc('signal')] = 'SignalNone'
          elif current_signal != previous_signal:
              if current_signal == 'SignalLong':
                  if previous_close < current_close:
                      df_filtered.iloc[i - 1, df_filtered.columns.get_loc('signal')] = 'SignalNone'
                      df_filtered.iloc[i, df_filtered.columns.get_loc('signal')] = 'SignalNone'
              elif current_signal == 'SignalShort':
                  if previous_close > current_close:
                      df_filtered.iloc[i - 1, df_filtered.columns.get_loc('signal')] = 'SignalNone'
                      df_filtered.iloc[i, df_filtered.columns.get_loc('signal')] = 'SignalNone'

      return df_filtered



    filter_1 = filter_pivots(df)

    df.update(filter_1)
    next_filter = df[['Close', 'signal']].copy()

    filter_2 = filter_pivots(next_filter)
    df.update(filter_2)

    previous_signal = None
    for i in range(len(df)):
        if df.iloc[i, filter_2.columns.get_loc('signal')] == "SignalNone" and previous_signal is not None:
            df.iloc[i, filter_2.columns.get_loc('signal')] = previous_signal 
        elif df.iloc[i, filter_2.columns.get_loc('signal')] != "SignalNone":
            previous_signal = df.iloc[i, filter_2.columns.get_loc('signal')]


    df_fixed = df.copy()
    df_fixed.loc[:,'signal'] = df_fixed.loc[:,'signal'].replace({'SignalLong': 1, 'SignalShort': 0})
    df_fixed = df_fixed.ffill()

    feature_names = [col for col in df_fixed.columns if col != 'signal']

    # Save feature names to a JSON file
    with open(f"models/{coin}_feature_names.json", 'w') as f:
        json.dump(feature_names, f)

    return df_fixed

    
def tt_split(df, timestep, coin):

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


    timestep = timestep
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
    joblib.dump(scaler_params, f"models/{coin}_scaler_params.joblib")


    # Assuming y_train is your labels tensor
    if isinstance(y_train, torch.Tensor):
        y_train_np = y_train.cpu().numpy()
    else:
        y_train_np = y_train

    unique_labels = np.unique(y_train_np)




    num_negatives = np.sum(y_train_np == 0)
    num_positives = np.sum(y_train_np == 1)
    pos_weight_value = 1.5 # num_negatives / num_positives
    # print(f"pos weight: {pos_weight_value}")
    
    # Create a tensor for pos_weight. Ensure this is a float tensor because it will be used in BCEWithLogitsLoss
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float)

    pos_weight = pos_weight.to('cpu')

    return x_train, x_test, y_train, y_test, pos_weight


def save_model(model, epoch, coin):
    model_path = f'models/{coin}_trained_model_lstm_{epoch}.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')
    
    
    
def plot_target(data):

    signal = data['signal']
    entries = signal == 2
    exits = signal == 0
    pf = vbt.Portfolio.from_signals(
        close=data.Close,
        long_entries=entries,
        short_entries=exits,
        size=100,
        size_type='value',
        init_cash='auto',
        accumulate=True
    )
    stats = pf.stats()
    vbt.settings.set_theme('dark')
    vbt.settings['plotting']['layout']['width'] = 1000
    vbt.settings['plotting']['layout']['height'] = 500
    pf.plot({"orders", "cum_returns"}).show()


def backtest(df, model, x_test):
    model.eval()
    with torch.no_grad():
        
        y_test_pred = model(x_test)
        probabilities = torch.softmax(y_test_pred, dim=1)
        _, predicted_labels = torch.max(probabilities, 1)
    predicted_labels_numpy = predicted_labels.numpy()


    df_split = df[-len(predicted_labels_numpy):].copy()
    df_split.loc[:, "signal"] = predicted_labels_numpy
    signal = df_split['signal']
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


