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
# - [x] TODO mike acorn data
# - [ ] TODO mike handle multiple houses. Multiindex

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
plt.rcParams['figure.figsize'] = (12.0, 4.0)
plt.style.use('ggplot')

from pathlib import Path
from tqdm.auto import tqdm

import pytorch_lightning as pl
# -

from seq2seq_time.data.dataset import Seq2SeqDataSet, Seq2SeqDataSets
from seq2seq_time.predict import predict

import logging, sys
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)

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

def get_smartmeter_df(indir=Path('../data/raw/smart-meters-in-london'), max_files=1):
    """
    Data loading and cleanding is always messy, so understand this code is optional.
    """
    
    # Load csv files
    csv_files = sorted((indir/'halfhourly_dataset').glob('*.csv'))[:max_files]
    
    # concatendate them
    df = pd.concat([pd.read_csv(f, parse_dates=[1], na_values=['Null']) for f in csv_files])
    
    # Add ACORN categories
    df_households = pd.read_csv(indir/'informations_households.csv')
    df_households = df_households[['LCLid', 'stdorToU', 'Acorn_grouped']]
    df = pd.merge(df, df_households, on='LCLid')
    
    df = df.set_index('tstp')
    
    # Drop nan and 0's
    df = df[df['energy(kWh/hh)']!=0]
    df = df.dropna()
    
    # Add time features 
    time = df.index.to_series()
    df["month"] = time.dt.month
    df['day'] = time.dt.day
    df['week'] = time.dt.week
    df['hour'] = time.dt.hour
    df['minute'] = time.dt.minute
    df['dayofweek'] = time.dt.dayofweek
    
    # Load weather data
    df_weather = pd.read_csv(indir/'weather_hourly_darksky.csv', parse_dates=[3])
    use_cols = ['visibility', 'windBearing', 'temperature', 'time', 'dewPoint',
           'pressure', 'apparentTemperature', 'windSpeed', 
           'humidity']
    df_weather = df_weather[use_cols].set_index('time')
    df_weather = df_weather.resample(freq).first().ffill()  # Resample to match energy data   
    
    # Join weather and energy data
    df = pd.merge(df, df_weather, how='inner', left_index=True, right_index=True, sort=True)
    
    # Holidays
    df_hols = pd.read_csv(indir/'uk_bank_holidays.csv', parse_dates=[0])
    holidays = set(df_hols['Bank holidays'].dt.round('D'))  
    def is_holiday(dt):
        return dt in holidays
    days = df.index.floor('D')
    holiday_mapping = days.unique().to_series().apply(is_holiday).astype(int).to_dict()
    df['holiday'] = days.to_series().map(holiday_mapping).values

    # Loop over houses
    for name, df_h in df.groupby('LCLid'):

        yield df_h


# -
# Our dataset is the london smartmeter data. But at half hour intervals

# +
dfs = get_smartmeter_df()

# Just get the first one for now
dfs = list(dfs)

# df = df.resample(freq).first().dropna() # Where empty we will backfill, this will respect causality, and mostly maintain the mean

# df = df.tail(int(max_rows)).copy() # Just use last X rows
df = pd.concat(dfs[:6], 0)
# df = dfs[0]
# -

df.LCLid.unique()



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

output_scaler = next(filter(lambda r:r[0][0] in columns_target, scaler.features))[-1]
output_scaler

dfs_norm = [d.resample(freq).first().ffill().dropna() for _, d in df_norm.groupby('LCLid')]
len(dfs_norm)

# +
# split data, with the test in the future
n_split = -int(len(dfs_norm)*0.2)
df_train = dfs_norm[:n_split]
df_test = dfs_norm[n_split:]

# Show split
pd.concat(df_train)['energy(kWh/hh)'].plot(label='train')
pd.concat(df_test)['energy(kWh/hh)'].plot(label='test')
plt.ylabel('energy(kWh/hh)')
plt.legend()
# -



# ### Dataset

# These are the columns that we wont know in the future
# We need to blank them out in x_future
columns_blank=['visibility',
       'windBearing', 'temperature', 'dewPoint', 'pressure',
       'apparentTemperature', 'windSpeed', 'humidity']

ds_train = Seq2SeqDataSets(df_train,
                          window_past=window_past,
                          window_future=window_future,
                          columns_blank=columns_blank)
ds_test = Seq2SeqDataSets(df_test,
                         window_past=window_past,
                         window_future=window_future,
                         columns_blank=columns_blank)
print(ds_train)
print(ds_test)

# we can treat it like an array
ds_train[0]
len(ds_train)
ds_train[-1]

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

class Seq2SeqLSTMDecoder(nn.Module):
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
# ## Lightning

# +
import pytorch_lightning as pl

class PL_Seq2Seq(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self._model = Seq2SeqLSTMDecoder(**hparams)

    def forward(self, x_past, y_past, x_future, y_future=None):
        """Eval/Predict"""
        y_dist = self._model(x_past, y_past, x_future)
        return y_dist

    def training_step(self, batch, batch_idx):
        x_past, y_past, x_future, y_future = batch
        y_dist = self.forward(*batch)
        loss = -y_dist.log_prob(y_future).mean()
        self.log_dict({'loss/train':loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x_past, y_past, x_future, y_future = batch
        y_dist = self.forward(*batch)
        loss = -y_dist.log_prob(y_future).mean()
        self.log_dict({'loss/val':loss})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


# -

from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import CSVLogger
from pl_bolts.callbacks import PrintTableMetricsCallback



# +
input_size = x_past.shape[-1]
output_size = y_future.shape[-1]

model = PL_Seq2Seq(input_size=input_size,
                   input_size_decoder=input_size,
                   output_size=output_size,
                   hidden_size=32,
                   lstm_layers=2,
                   lstm_dropout=0.25).to(device)

logger = CSVLogger("logs", name="seq2seq")
trainer = pl.Trainer(gpus=1,
                     logger=logger)
dl_train = DataLoader(ds_train,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=8)
dl_test = DataLoader(ds_test, batch_size=batch_size, num_workers=4)
trainer.fit(model, dl_train, dl_test)
# -
df_hist = pd.read_csv(trainer.logger.experiment.metrics_file_path)
df_hist['epoch'] = df_hist['epoch'].ffill()
df_histe = df_hist.set_index('epoch').groupby('epoch').mean()
df_histe[['loss/train', 'loss/val']].plot()
df_histe

# ## Predict
#

ds_preds = predict(model.to(device), ds_test.datasets[0], batch_size, device=device, scaler=output_scaler)
ds_preds


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
    
    
    plt.figure(figsize=(12, 4))
    
    plt.scatter(xf, yt, label='true', c='k', s=6)
    ylim = plt.ylim()

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
    
    # plot a red line for now
    plt.vlines(x=now, ymin=0, ymax=1, label='now', color='r')
    plt.ylim(*ylim)

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
ds_preds.mean('t_source').plot.scatter('t_ahead_hours', 'nll') # Mean over all predictions

# Tidy the graph
n = len(ds_preds.t_source)
plt.ylabel('Negative Log Likelihood (lower is better)')
plt.xlabel('Hours ahead')
plt.title(f'NLL vs time (no. samples={n})')
# -



# Make a plot of the NLL over time. Does this solution get worse with time?
d = ds_preds.mean('t_ahead').groupby('t_source').mean().plot.scatter('t_source', 'nll')
plt.xticks(rotation=45)
plt.title('NLL over time (lower is better)')
1

# A scatter plot is easy with xarray
ds_preds.plot.scatter('y_true', 'y_pred', s=.01)








