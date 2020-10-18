# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: seq2seq-time
#     language: python
#     name: seq2seq-time
# ---

# # Sequence to Sequence Models for Timeseries Regression
#
#
# In this notebook we are going to tackle a harder problem: 
# - predicting the future on a timeseries
# - using an LSTM
# - with rough uncertainty (uncalibrated)
# - outputing sequence of predictions
#
# <img src="../reports/figures/Seq2Seq for regression.png" />
#
#

#
# - [ ] TODO mike autocorrelation baseline
# - [ ] TODO mike acorn data

# OPTIONAL: Load the "autoreload" extension so that code can change. But blacklist large modules
# %load_ext autoreload
# %autoreload 2
# %aimport -pandas
# %aimport -torch
# %aimport -numpy
# %aimport -matplotlib
# %aimport -dask
# %aimport -tqdm
# %matplotlib inline

# +
# Imports
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import torch
import torch.utils.data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm.auto import tqdm

import pytorch_lightning as pl
# -

from seq2seq_time.data.dataset import Seq2SeqDataSet
from seq2seq_time.predict import predict

import logging, sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# ## Parameters

# +
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'using {device}')

columns_target=['energy(kWh/hh)']
window_past = 48*4
window_future = 48*4
batch_size = 64
num_workers = 0
freq = '30T'
max_rows = 1e5


# -

# ## Load data

# +

def get_smartmeter_df(indir=Path('../data/raw/smart-meters-in-london')):
    """
    Data loading and cleanding is always messy, so understand this code is optional.
    """
    
    # Load csv files
    csv_files = sorted((indir/'halfhourly_dataset').glob('*.csv'))[:1]
    
#     import pdb; pdb.set_trace() # you can use debugging in jupyter to interact with variables inside a function
    
    # concatendate them
    df = pd.concat([pd.read_csv(f, parse_dates=[1], na_values=['Null']) for f in csv_files])
    
    # Add ACORN categories
    df_households = pd.read_csv(indir/'informations_households.csv')
    df_households = df_households[['LCLid', 'stdorToU', 'Acorn_grouped']]
    df = pd.merge(df, df_households, on='LCLid')

    # Take the mean over all houses
    name, df = next(iter(df.groupby('LCLid')))
    df = df.set_index('tstp')
    print(df)

    # Load weather data
    df_weather = pd.read_csv(indir/'weather_hourly_darksky.csv', parse_dates=[3])
    use_cols = ['visibility', 'windBearing', 'temperature', 'time', 'dewPoint',
           'pressure', 'apparentTemperature', 'windSpeed', 
           'humidity']
    df_weather = df_weather[use_cols].set_index('time')
    df_weather = df_weather.resample(freq).first().ffill()  # Resample to match energy data    

    # Join weather and energy data
    df = pd.concat([df, df_weather], 1).dropna()    
    
    # Also find bank holidays
    df_hols = pd.read_csv(indir/'uk_bank_holidays.csv', parse_dates=[0])
    holidays = set(df_hols['Bank holidays'].dt.round('D'))  

    time = df.index.to_series()
    def is_holiday(dt):
        return dt.floor('D') in holidays
    df['holiday'] = time.apply(is_holiday).astype(int)
    
    # TODO pd.read_csv('../data/raw/smart-meters-in-london/acorn_details.csv', engine='python')


    # Add time features    
    df["month"] = time.dt.month
    df['day'] = time.dt.day
    df['week'] = time.dt.week
    df['hour'] = time.dt.hour
    df['minute'] = time.dt.minute
    df['dayofweek'] = time.dt.dayofweek

    # Drop nan and 0's
    df = df[df['energy(kWh/hh)']!=0]
    df = df.dropna()

    # sort by time
    df = df.sort_index()
    
    return df
# -
# Our dataset is the london smartmeter data. But at half hour intervals

# +
df = get_smartmeter_df()

# df = df.resample(freq).first().dropna() # Where empty we will backfill, this will respect causality, and mostly maintain the mean

df = df.tail(int(max_rows)).copy() # Just use last X rows
df
# -

df.describe()

# +
import sklearn
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn_pandas import DataFrameMapper

columns_input_numeric = list(df.drop(columns=columns_target)._get_numeric_data().columns)
columns_categorical = list(set(df.columns)-set(columns_input_numeric)-set(columns_target))

output_scalers = [([n], StandardScaler()) for n in columns_target]
transformers=output_scalers + \
[([n], StandardScaler()) for n in columns_input_numeric] + \
[([n], OrdinalEncoder()) for n in columns_categorical]
scaler = DataFrameMapper(transformers, df_out=True)
df_norm = scaler.fit_transform(df)
df_norm
# -

output_scaler = next(filter(lambda r:r[0][0] in columns_target, mapper4.features))[-1]
output_scaler

# # Resample
df_norm = df_norm.resample(freq).first().fillna(0)

# +
# split data, with the test in the future
n_split = -int(len(df)*0.2)
df_train = df_norm[:n_split]
df_test = df_norm[n_split:]

# Show split
df_train['energy(kWh/hh)'].plot(label='train')
df_test['energy(kWh/hh)'].plot(label='test')
plt.ylabel('energy(kWh/hh)')
plt.legend()
# -
df_norm


columns_blank=['visibility',
       'windBearing', 'temperature', 'dewPoint', 'pressure',
       'apparentTemperature', 'windSpeed', 'humidity']

ds_train = Seq2SeqDataSet(df_train,
                          window_past=window_past,
                          window_future=window_future,
                          columns_blank=columns_blank)
ds_test = Seq2SeqDataSet(df_test,
                         window_past=window_past,
                         window_future=window_future,
                         columns_blank=columns_blank)
print(ds_train)
print(ds_test)

# %%timeit
for i in range(100):
    ds_train[i]

# we can treat it like an array
ds_train[0]
len(ds_train)
ds_train[0][2][-2]

# +
# We can get rows
x_past, y_past, x_future, y_future = ds_train.get_rows(10)

# Plot one instance, this is what the model sees
y_past['energy(kWh/hh)'].plot(label='past')
y_future['energy(kWh/hh)'].plot(ax=plt.gca(), label='future')
plt.legend()
plt.ylabel('energy(kWh/hh)')

# Notice we've added on two new columns tsp (time since present) and is_past
x_past.tail()
# -

# Notice we've hidden some future columns to prevent cheating
x_future.tail()


# ## Model

# +

class Seq2SeqNet(nn.Module):
    def __init__(self, input_size, input_size_decoder, output_size, hidden_size=32, lstm_layers=2, lstm_dropout=0, _min_std = 0.05):
        super().__init__()
        self._min_std = _min_std

        self.encoder = nn.LSTM(
            input_size=input_size + output_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
        )
        self.decoder = nn.LSTM(
            input_size=input_size_decoder,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
        )
        self.mean = nn.Linear(hidden_size, output_size)
        self.std = nn.Linear(hidden_size, output_size)

    def forward(self, context_x, context_y, target_x, target_y=None):
        x = torch.cat([context_x, context_y], -1)
        _, (h_out, cell) = self.encoder(x)
        
        ## Shape
        # hidden = [batch size, n layers * n directions, hid dim]
        # cell = [batch size, n layers * n directions, hid dim]
        # output = [batch size, seq len, hid dim * n directions]
        outputs, (_, _) = self.decoder(target_x, (h_out, cell))
        
        
        # outputs: [B, T, num_direction * H]
        mean = self.mean(outputs)
        log_sigma = self.std(outputs)
        log_sigma = torch.clamp(log_sigma, np.log(self._min_std), -np.log(self._min_std))

        sigma = torch.exp(log_sigma)
        y_dist = torch.distributions.Normal(mean, sigma)
        return y_dist


# -



# +
input_size = x_past.shape[-1]
output_size = y_future.shape[-1]

model = Seq2SeqNet(input_size, input_size, output_size,
                   hidden_size=32, 
                   lstm_layers=2, 
                   lstm_dropout=0).to(device)
model
# -
# Init the optimiser
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# +

past_x = torch.rand((batch_size, window_past, input_size)).to(device)
future_x = torch.rand((batch_size, window_future, input_size)).to(device)
past_y = torch.rand((batch_size, window_past, output_size)).to(device)
future_y = torch.rand((batch_size, window_future, output_size)).to(device)
output = model(past_x, past_y, future_x, future_y)  
print(output)

from torchsummaryX import summary
summary(model, past_x, past_y, future_x, future_y )
1
# -

# ## Training





# +
def train_epoch(ds, model, bs=128):
    model.train()

    training_loss = []

    # Put data into a torch loader
    load_train = torch.utils.data.dataloader.DataLoader(
        ds,
        batch_size=bs,
        pin_memory=False,
        num_workers=num_workers,
        shuffle=True,
    )

    for batch in tqdm(load_train, leave=False, desc='train'):
        # Send data to gpu
        x_past, y_past, x_future, y_future = [d.to(device) for d in batch]

        # Discard previous gradients
        optimizer.zero_grad()
        
        # Run model
        y_dist = model(x_past, y_past, x_future, y_future)
        
        # Get loss, it's Negative Log Likelihood
        loss = -y_dist.log_prob(y_future).mean()

        # Backprop
        loss.backward()
        optimizer.step()

        # Record stats
        training_loss.append(loss.item())

    return np.mean(training_loss)


def test_epoch(ds, model, bs=512):
    model.eval()

    test_loss = []
    load_test = torch.utils.data.dataloader.DataLoader(ds,
                                                       batch_size=bs,
                                                       pin_memory=False,
                                                       num_workers=num_workers)
    for batch in tqdm(load_test, leave=False, desc='test'):
        # Send data to gpu
        x_past, y_past, x_future, y_future = [d.to(device) for d in batch]
        with torch.no_grad():
            # Run model
            y_dist = model(x_past, y_past, x_future, y_future)
            # Get loss, it's Negative Log Likelihood
            loss = -y_dist.log_prob(y_future).mean()

        test_loss.append(loss.item())

    return np.mean(test_loss)


def training_loop(ds_train, ds_test, model, epochs=1, bs=128):
    all_losses = []
    try:
        test_loss = test_epoch(ds_test, model)
        print(f"Start: Test Loss = {test_loss:.2f}")
        for epoch in tqdm(range(epochs), desc='epochs'):
            loss = train_epoch(ds_train, model, bs=bs)
            print(f"Epoch {epoch+1}/{epochs}: Training Loss = {loss:.2f}")

            test_loss = test_epoch(ds_test, model)
            print(f"Epoch {epoch+1}/{epochs}: Test Loss = {test_loss:.2f}")
            print("-" * 50)

            all_losses.append([loss, test_loss])

    except KeyboardInterrupt:
        # This lets you stop manually. and still get the results
        pass

    # Visualising the results
    all_losses = np.array(all_losses)
    plt.plot(all_losses[:, 0], label="Training")
    plt.plot(all_losses[:, 1], label="Test")
    plt.title("Loss")
    plt.legend()

    return all_losses


# -

# this might take 1 minute per epoch on a gpu
training_loop(ds_train, ds_test, model, epochs=8, bs=batch_size)
1

# ## Predict
#

# TODO get working
output_scaler = scaler.transformers[-4][1]
ds_preds = predict(model, ds_test, batch_size*6, device=device, scaler=output_scaler)



# +
# TODO Metrics... smape etc

# +
def plot_prediction(ds_preds, i):
    """Plot a prediction into the future, at a single point in time."""
    d = ds_preds.isel(t_source=i)

    # Get arrays
    xf = d.t_target
    yp = d.y_pred
    s = d.y_pred_std
    yt = d.y_true
    now = d.t_source.squeeze()

    # plot prediction
    plt.fill_between(xf, yp-2*s, yp+2*s, alpha=0.25,
            facecolor="b",
            interpolate=True,
            label="2 std",)
    plt.plot(xf, yp, label='pred', c='b')

    # plot true
    plt.scatter(
        d.t_past,
        d.y_past,
        c='k',
        s=6
    )
    plt.scatter(xf, yt, label='true', c='k', s=6)
    
    # plot a red line for now
    plt.vlines(x=now, ymin=0, ymax=1, label='now', color='r')

    now=pd.Timestamp(now.values)
    plt.title(f'Prediction NLL={d.nll.mean().item():2.2g}')
    plt.xlabel(f'{now.date()}')
    plt.ylabel('energy(kWh/hh)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()
    
# plot_prediction(ds_preds, 0) 
# plot_prediction(ds_preds, 12) # 6 hours later
plot_prediction(ds_preds, 24) # 12 hours later
plot_prediction(ds_preds, 48) # 12 hours later
# -

# ## Error vs time ahead



# +
d = ds_preds.mean('t_source') # Mean over all predictions

# Plot with xarray, it has a pandas like interface
d.plot.scatter('t_ahead_hours', 'nll')

# Tidy the graph
n = len(ds_preds.t_source)
plt.ylabel('Negative Log Likelihood (lower is better)')
plt.xlabel('Hours ahead')
plt.title(f'NLL vs time (no. samples={n})')
# -

d = ds_preds.mean('t_source') # Mean over all predictions
d['likelihood'] = np.exp(-d.nll) # get likelihood, after taking mean in log domain
d.plot.scatter('t_ahead_hours', 'likelihood')



# Make a plot of the NLL over time. Does this solution get worse with time?
# this is hard because we need to take the mean over t_ahead
# then group by t_source
d = ds_preds.mean('t_ahead').groupby('t_source').mean()
# And even then it's clearer with smoothing
d.plot.scatter('t_source', 'nll')
plt.xticks(rotation=45)
plt.title('NLL over time (lower is better)')
1

# A scatter plot is easy with xarray
ds_preds.plot.scatter('y_true', 'y_pred', s=.01)


