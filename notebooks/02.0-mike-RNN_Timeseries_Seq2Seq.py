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
# https://medium.com/@boitemailjeanmid/smart-meters-in-london-part1-description-and-first-insights-jean-michel-d-db97af2de71b
#

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
plt.rcParams['figure.figsize'] = (12.0, 3.0)
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
window_past = 48*2
window_future = 48*2
batch_size = 256
num_workers = 5
freq = '30T'
max_rows = 2e5


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
    
    
    df = df.sort_values(['tstp', 'LCLid'])
    
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
    
    # sort
    df = df.reset_index().sort_values(['LCLid', 'index']).set_index('index')
    df.index.name = 'Date'
    
    return df
# -
# Our dataset is the london smartmeter data. But at half hour intervals

# +
df = get_smartmeter_df()

# # Just get the first one for now
# dfs = list(dfs)

# # df = df.resample(freq).first().dropna() # Where empty we will backfill, this will respect causality, and mostly maintain the mean

df = df.tail(int(max_rows)).copy() # Just use last X rows
# df = pd.concat(dfs[:6], 0)
# # df = dfs[0]
df.LCLid.value_counts()
# -



# ### Plot/explore

df



# +
import holoviews as hv
from holoviews import opts

from holoviews.plotting.links import RangeToolLink

import datashader as ds

from holoviews.operation.datashader import datashade, shade, dynspread, rasterize
from holoviews.operation import decimate

hv.extension('bokeh')


def house_curve(Name=None):
    if isinstance(Name, int):
        name = df.LCLid.unique()[Name]
    d = df[df.LCLid == Name]
    d_curve = hv.Curve(d, 'Date', 'energy(kWh/hh)', label=Name)
    return d_curve


dmap = hv.DynamicMap(house_curve, kdims=['Name'])
dmap = dmap.redim.values(Name=list(df.LCLid.unique()))
dynspread(datashade(dmap).opts(width=800,
                     height=300,
                     tools=['xwheel_zoom', 'pan'],
                     active_tools=['xwheel_zoom', 'pan'],
                     default_tools=['reset', 'save', 'hover']
                    ))
# -




# ### Profiling

# +
# from pandas_profiling import ProfileReport
# profile = ProfileReport(df, title="Pandas Profiling Report", minimal=True)
# profile
# -

# ### Norm

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

# +
# split data, with the test in the future

d0 =df_norm.index.min()
d1 = df_norm.index.max()
split_time = d0+(d1-d0)*0.8
split_time = split_time.round('1D')
print(split_time)
df_train = df_norm.groupby('LCLid').apply(lambda d:d.loc[:split_time]).reset_index(level=0, drop=True)
df_test = df_norm.groupby('LCLid').apply(lambda d:d.loc[split_time:]).reset_index(level=0, drop=True)
# df_test
# -

# Show split
df_train['energy(kWh/hh)'].plot(label='train')
df_test['energy(kWh/hh)'].plot(label='test')
plt.ylabel('energy(kWh/hh)')
plt.legend()
# ### Dataset
# These are the columns that we wont know in the future
# We need to blank them out in x_future
columns_blank=['visibility',
       'windBearing', 'temperature', 'dewPoint', 'pressure',
       'apparentTemperature', 'windSpeed', 'humidity']
df_trains = [d.resample(freq).first().ffill().dropna() for _,d in df_train.groupby('LCLid')]
df_tests = [d.resample(freq).first().ffill().dropna() for _,d in df_test.groupby('LCLid')]
ds_train = Seq2SeqDataSets(df_trains,
                          window_past=window_past,
                          window_future=window_future,
                          columns_blank=columns_blank)
ds_test = Seq2SeqDataSets(df_tests,
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

# ## Plot helpers

from seq2seq_time.models.lstm_seq2seq import LSTMSeq2Seq
from seq2seq_time.models.lstm import LSTM
from seq2seq_time.models.baseline import BaselineLast
from seq2seq_time.models.transformer import Transformer
from seq2seq_time.models.transformer_seq2seq import TransformerSeq2Seq
# ## Plots
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
    
def plot_performance(ds_preds, full=False):
    """Multiple plots using xr_preds"""
    plot_prediction(ds_preds, 24)
    plot_prediction(ds_preds, 480)

    ds_preds.mean('t_source').plot.scatter('t_ahead_hours', 'nll') # Mean over all predictions
    n = len(ds_preds.t_source)
    plt.ylabel('Negative Log Likelihood (lower is better)')
    plt.xlabel('Hours ahead')
    plt.title(f'NLL vs time ahead (no. samples={n})')
    plt.show()

    # Make a plot of the NLL over time. Does this solution get worse with time?
    if full:
        d = ds_preds.mean('t_ahead').groupby('t_source').mean().plot.scatter('t_source', 'nll')
        plt.xticks(rotation=45)
        plt.title('NLL over source time (lower is better)')
        plt.show()

    # A scatter plot is easy with xarray
    if full:
        plt.figure(figsize=(5, 5))
        ds_preds.plot.scatter('y_true', 'y_pred', s=.01)
        plt.show()
    
    print(f'mean_NLL {ds_preds.nll.mean().item():2.2f}')
# -



def plot_hist(trainer):
    try:
        df_hist = pd.read_csv(trainer.logger.experiment.metrics_file_path)
        df_hist['epoch'] = df_hist['epoch'].ffill()
        df_histe = df_hist.set_index('epoch').groupby('epoch').mean()
        if len(df_histe)>1:
            df_histe[['loss/train', 'loss/val']].plot(title='history')
        return df_histe
    except Exception:
        pass


# ## Lightning

# +
import pytorch_lightning as pl

class PL_MODEL(pl.LightningModule):
    def __init__(self, model, lr=3e-4, patience=2):
        super().__init__()
        self._model = model
        self.lr = lr
        self.patience = patience

    def forward(self, x_past, y_past, x_future, y_future=None):
        """Eval/Predict"""
        y_dist = self._model(x_past, y_past, x_future)
        return y_dist

    def training_step(self, batch, batch_idx, phase='train'):
        x_past, y_past, x_future, y_future = batch
        y_dist = self.forward(*batch)
        loss = -y_dist.log_prob(y_future).mean()
        self.log_dict({f'loss/{phase}':loss})
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, phase='val')

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            patience=self.patience,
            verbose=True,
            min_lr=1e-7,
        )
        return {'optimizer': optim, 'lr_scheduler': scheduler, 'monitor': 'loss/val'}


# -

# # Run
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import CSVLogger
from pl_bolts.callbacks import PrintTableMetricsCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


# +
# Init data
x_past, y_past, x_future, y_future = ds_train.get_rows(10)
input_size = x_past.shape[-1]
output_size = y_future.shape[-1]

dl_train = DataLoader(ds_train,
                      batch_size=batch_size,
                      shuffle=True,
                      pin_memory=num_workers==0,
                      num_workers=num_workers)
dl_test = DataLoader(ds_test, batch_size=batch_size, num_workers=num_workers)
# -

# Baseline model
pt_model = BaselineLast()
model = PL_MODEL(pt_model).to(device)
trainer = pl.Trainer(gpus=1,
                     max_epochs=1, 
                     limit_train_batches=0.01,
                     logger=CSVLogger("logs",
                                      name=type(pt_model).__name__),
                    )
trainer.fit(model, dl_train, dl_test)
print(plot_hist(trainer))
ds_preds = predict(model.to(device), ds_test.datasets[0], batch_size, device=device, scaler=output_scaler)
print(f'baseline nll: {ds_preds.nll.mean().item():2.2g}')



models = [
#     BaselineLast(),
    LSTM(input_size,
         output_size,
         hidden_size=80,
         lstm_layers=3,
         lstm_dropout=0.3),
    Transformer(input_size,
                output_size,
                attention_dropout=0.3,
                nhead=8,
                nlayers=6,
                hidden_size=64),
    LSTMSeq2Seq(input_size,
                output_size,
                hidden_size=64,
                lstm_layers=2,
                lstm_dropout=0.25),
    TransformerSeq2Seq(input_size,
                       output_size,
                       hidden_size=64,
                       nhead=8,
                       nlayers=4,
                       attention_dropout=0.3),
#     Transformer(input_size,
#                 output_size,
#                 attention_dropout=0.2,
#                 nhead=8,
#                 nlayers=6,
#                 hidden_size=128),
#     LSTM(input_size,
#          output_size,
#          hidden_size=128,
#          lstm_layers=3,
#          lstm_dropout=0.3),
]





for pt_model in models:
    name = type(pt_model).__name__
    print(name)

    # Wrap in lightning
    patience = 2
    model = PL_MODEL(pt_model, patience=patience, lr=3e-4).to(device)

    # Trainer
    
    trainer = pl.Trainer(gpus=1,
                         min_epochs=1,
                         max_epochs=10,
                         amp_level='O1',
                         precision=16,
                         gradient_clip_val=0.5,
                         logger=CSVLogger("logs",
                                          name=type(pt_model).__name__),
                         callbacks=[
                             EarlyStopping(monitor='loss/val', patience=patience*2),
                             PrintTableMetricsCallback()
                         ],
    )

    # Train
    trainer.fit(model, dl_train, dl_test)

    # Performance
    print(plot_hist(trainer))

    ds_preds = predict(model.to(device),
                       ds_test.datasets[0],
                       batch_size,
                       device=device,
                       scaler=output_scaler)
    plot_performance(ds_preds)



# # holoviews pred

import holoviews as hv
from holoviews import opts


# +
def plot_prediction_now(t_source):
    """Plot predictions with holoviews"""

    # Let us pass in an int
    if isinstance(t_source, int):
        t_source = ds_preds.t_source[t_source].to_pandas()

    d = ds_preds.sel(t_source=t_source)

    # Sometimes there are duplicate time, take the first
    if len(d.t_source.shape) and d.t_source.shape[0] > 0:
        d = d.isel(t_source=0)
    if len(d.t_source.shape) and d.t_source.shape[0] == 0:
        return None

    now = d.t_source.to_pandas()

    # Plot true
    x = np.concatenate([d.t_past, d.t_target])
    yt = np.concatenate([d.y_past, d.y_true])
    p = hv.Scatter({
        'x': x,
        'y': yt
    }, label='true').opts(color='black', framewise=True)

    # Get arrays
    xf = d.t_target.values
    yp = d.y_pred
    s = d.y_pred_std
    p *= hv.Curve({
        'x': xf,
        'y': yp
    }, label='pred').opts(color='blue')
    p *= hv.Area((xf, yp - 2 * s, yp + 2 * s),
                 vdims=['y', 'y2'],
                 label='2*std').opts(alpha=0.5, line_width=0)

    # plot now line
    p *= hv.VLine(now, label='now').opts(color='red')
    return p.opts(title=f'Prediction at {now}. NLL={d.nll.mean().item():2.2f}')


dmap_pred = (hv.DynamicMap(plot_prediction_now, kdims=['t_source'])
        .redim.values(t_source=ds_preds.t_source.to_pandas())
        .opts(width=800,
                     height=300, 
                    ))
dmap_pred


# +
def plot_predictions_vs_time(it_ahead):
    """Plot predictions vs time with holoviews"""

    d = ds_preds.isel(t_ahead=it_ahead).groupby('t_source').first()

    p = hv.Scatter({
        'x': d.t_source,
        'y': d.y_true
    }, label='true').opts(color='black')

    # Get arrays
    xf = d.t_source.values
    yp = d.y_pred
    s = d.y_pred_std
    p *= hv.Curve({
        'x': xf,
        'y': yp
    }, label='pred').opts(color='blue')
    p *= hv.Area((xf, yp - 2 * s, yp + 2 * s),
                 vdims=['y', 'y2'],
                 label='2*std').opts(alpha=0.5, line_width=0)


    return p.opts(title=f'Prediction at {it_ahead * pd.Timedelta(freq)} ahead. NLL={d.nll.mean().item():2.2f}')


dmap_preds = (hv.DynamicMap(plot_predictions_vs_time, kdims=['it_ahead'])
        .redim.values(it_ahead=range(ds_preds.t_ahead.shape[0]))
        .opts(width=800,
                     height=300, 
                    ))
dmap_preds
# plot_prediction2(10).opts(width=800)
# -

d = ds_preds.mean('t_source')['nll'].groupby('t_ahead_hours').mean()
nll_vs_tahead = hv.Curve((d.t_ahead_hours, d)).redim(x='hours ahead', y='nll').opts(width=800)
nll_vs_tahead

d = ds_preds.mean('t_ahead')['nll'].groupby('t_source').mean()
nll_vs_time = hv.Curve(d).opts(width=800)
nll_vs_time

true_vs_pred = hv.Scatter((ds_preds.y_true, ds_preds.y_pred))
dynspread(datashade(true_vs_pred))

# # Summarize experiments

# # LR finder

# +

# Run learning rate finder
lr_finder = trainer.tuner.lr_find(model)

# Results can be found in
lr_finder.results

# Plot with
fig = lr_finder.plot(suggest=True)
fig.show()

# Pick point based on plot, or get suggestion
new_lr = lr_finder.suggestion()
# -


