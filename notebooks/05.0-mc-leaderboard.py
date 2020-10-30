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
# - by outputing sequence of predictions
# - with rough uncertainty (uncalibrated)
# - using forecasted information (like weather report, week, or cycle of the moon)
#
# Not many papers benchmark movels for multivariate regression, much less seq prediction with uncertainty. So this notebook will try a range of models on a range of dataset.
#
# We do this using a sequence to sqequence interface
#
# <img src="../reports/figures/Seq2Seq for regression.png" />
#

# - [ ] tensorboard / wandb
# - [ ] show test train
# - [ ] val
# - [ ] don't overfit
# - [ ] TCN

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

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 2.0)
plt.style.use('ggplot')

from pathlib import Path
from tqdm.auto import tqdm

import pytorch_lightning as pl
# -

import warnings
warnings.simplefilter('once')

from seq2seq_time.data.dataset import Seq2SeqDataSet, Seq2SeqDataSets
from seq2seq_time.predict import predict, predict_multi
from seq2seq_time.util import dset_to_nc

import logging, sys
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# +
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import datashade, dynspread
hv.extension('bokeh')

# holoview datashader timeseries options
# %opts RGB [width=800 height=200 active_tools=["xwheel_zoom"] default_tools=["xpan","xwheel_zoom", "reset"] toolbar="right"]
# -


import warnings
warnings.filterwarnings("ignore")

# ## Parameters

# +
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'using {device}')

columns_target=['energy(kWh/hh)']
window_past = 48*2
window_future = 48*2
batch_size = 128
num_workers = 5
freq = '30T'
max_rows = 5e5
datasets_root = Path('../data/processed/')
window_past
# -

# ## Plot helpers



# +
def plot_prediction(ds_preds, i, ax=None, title='', std=False, label='pred', legend=False):
    """Plot a prediction into the future, at a single point in time."""    
    d = ds_preds.isel(t_source=i)

    # Get arrays
    xf = d.t_target
    yp = d.y_pred
    s = d.y_pred_std
    yt = d.y_true
    now = d.t_source.squeeze()
    
    plt.scatter(xf, yt, c='k', s=6, label='true' if legend else None)
    ylim = plt.ylim()

    # plot prediction
    if std:
        plt.fill_between(xf, yp-2*s, yp+2*s, alpha=0.25,
                facecolor="b",
                interpolate=True,
                label="2 std" if legend else None,)
    plt.plot(xf, yp, label=label)

    # plot true
    plt.scatter(
        d.t_past,
        d.y_past,
        c='k',
        s=6
    )
    
    # plot a red line for now
    plt.vlines(x=now, ymin=ylim[0], ymax=ylim[1], color='grey', ls='--')
    plt.ylim(*ylim)

    now=pd.Timestamp(now.values)
    plt.title(title or f'Prediction NLL={d.nll.mean().item():2.2g}')
    plt.xticks(rotation=0)    
    if legend:
        plt.legend()
    plt.xlabel(f'{now}')
    plt.ylabel(ds_preds.attrs['targets'])
    return now

def plot_performance(ds_preds, full=False):
    """Multiple plots using xr_preds"""
    plot_prediction(ds_preds, 24, std=True, legend=True)
    plt.show()

    ds_preds.mean('t_source').plot.scatter('t_ahead_hours', 'nll') # Mean over all predictions
    n = len(ds_preds.t_source)
    plt.ylabel('NLL (lower is better)')
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


# -
def plot_hist(trainer):
    try:
        df_hist = pd.read_csv(trainer.logger.experiment.metrics_file_path)
        df_hist['epoch'] = df_hist['epoch'].ffill()
        df_histe = df_hist.set_index('epoch').groupby('epoch').mean()
        if len(df_histe)>1:
            df_histe[['loss/train', 'loss/val']].plot(title='history')
        plt.show()
        return df_histe
    except Exception:
        pass

# ## Datasets



# +
from seq2seq_time.data.data import IMOSCurrentsVel, AppliancesEnergyPrediction, BejingPM25, GasSensor, MetroInterstateTraffic

datasets = [BejingPM25, GasSensor, AppliancesEnergyPrediction, MetroInterstateTraffic, IMOSCurrentsVel]
datasets
# -

# View train, test, val splits
l = hv.Layout()
for dataset in datasets:
    d = dataset(datasets_root)
    
    p = dynspread(
        datashade(hv.Scatter(d.df_train[d.columns_target[0]]),
                  cmap='red'))
    p *= dynspread(
        datashade(hv.Scatter(d.df_val[d.columns_target[0]]),
                  cmap='green'))
    p *= dynspread(
        datashade(hv.Scatter(d.df_test[d.columns_target[0]]),
                  cmap='blue'))
    p = p.opts(title=f"{dataset}")
    l += p
l.cols(1)

# ## Lightning

# +
import pytorch_lightning as pl

class PL_MODEL(pl.LightningModule):
    def __init__(self, model, lr=3e-4, patience=None, weight_decay=0):
        super().__init__()
        self._model = model
        self.lr = lr
        self.patience = patience
        self.weight_decay = weight_decay

    def forward(self, x_past, y_past, x_future, y_future=None):
        """Eval/Predict"""
        y_dist, extra = self._model(x_past, y_past, x_future, y_future)
        assert torch.isfinite(y_dist.loc).all(), 'output should be finite'
        return y_dist, extra

    def training_step(self, batch, batch_idx, phase='train'):
        x_past, y_past, x_future, y_future = batch
        y_dist, extra = self.forward(*batch)
        loss = -y_dist.log_prob(y_future).mean()
        assert torch.isfinite(loss).all(), 'loss should be finite'
        self.log_dict({f'loss/{phase}':loss})
        if ('loss' in extra) and (phase=='train'):
            # some models have a special loss
            loss = extra['loss']
            self.log_dict({f'model_loss/{phase}':loss})
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, phase='val')
    
    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr,  weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            patience=self.patience,
            verbose=True,
            min_lr=1e-7,
        ) if self.patience else None
        return {'optimizer': optim, 'lr_scheduler': scheduler, 'monitor': 'loss/val'}


# -

# # Run
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


# Models
from seq2seq_time.models.lstm_seq2seq import LSTMSeq2Seq
from seq2seq_time.models.lstm_seq import LSTMSeq
from seq2seq_time.models.lstm import LSTM
from seq2seq_time.models.baseline import BaselineLast
from seq2seq_time.models.transformer import Transformer
from seq2seq_time.models.transformer_autor import TransformerAutoR
from seq2seq_time.models.transformer_seq2seq import TransformerSeq2Seq
from seq2seq_time.models.transformer_seq import TransformerSeq
from seq2seq_time.models.neural_process import RANP
from seq2seq_time.models.transformer_process import TransformerProcess
from seq2seq_time.models.tcn import TemporalConvNet
# ## Plots
# +
import gc

def free_mem():
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


# -

models = [
    lambda: BaselineLast(),
#     lambda: TransformerAutoR(input_size,
#         output_size, hidden_out_size=32),
    lambda: RANP(input_size,
        output_size, hidden_dim=64, dropout=0.5, 
         latent_dim=32, n_decoder_layers=4),
    lambda: LSTM(input_size,
         output_size,
         hidden_size=32,
         lstm_layers=3,
         lstm_dropout=0.4),
    lambda: LSTMSeq2Seq(input_size,
                output_size,
                hidden_size=64,
                lstm_layers=2,
                lstm_dropout=0.4),
    lambda: TransformerSeq2Seq(input_size,
                       output_size,
                       hidden_size=64,
                       nhead=8,
                       nlayers=4,
                       attention_dropout=0.4),
    lambda: Transformer(input_size,
                output_size,
                attention_dropout=0.4,
                nhead=8,
                nlayers=6,
                hidden_size=64),
    lambda :TransformerProcess(input_size,
                output_size, hidden_size=16,
        latent_dim=8, dropout=0.5,
        nlayers=4,)
#     lambda :TemporalConvNet()
]
# models



# +
# GasSensor(datasets_root)
# -

# ## Train

from collections import defaultdict
results = defaultdict(dict)

# +
# tmp
model = Transformer(input_size,
                output_size,
                attention_dropout=0.4,
                nhead=2,
                nlayers=4,
                hidden_size=16)

x_past, y_past, x_future, y_future = next(iter(dl_val))
model(x_past, y_past, x_future, y_future)
# -

from seq2seq_time.metrics import rmse, smape

# +
for Dataset in datasets:
    dataset_name = Dataset.__name__
    dataset = Dataset(datasets_root)
    ds_train, ds_val, ds_test = dataset.to_datasets(window_past=window_past,
                                            window_future=window_future)

    # Init data
    x_past, y_past, x_future, y_future = ds_train.get_rows(10)
    input_size = x_past.shape[-1]
    output_size = y_future.shape[-1]

    # Loaders
    dl_train = DataLoader(ds_train,
                          batch_size=batch_size,
                          shuffle=True,
                          pin_memory=num_workers == 0,
                          num_workers=num_workers)
    dl_val = DataLoader(ds_val,
                         batch_size=batch_size,
                         num_workers=num_workers)

    for m_fn in models:
        try:
            free_mem()
            pt_model = m_fn()
            model_name = type(pt_model).__name__
            print(dataset_name, model_name)

            # Wrap in lightning
            patience = 3
            model = PL_MODEL(pt_model,
                             lr=3e-3, patience=patience,
                             weight_decay=1e-5).to(device)

            # Trainer            
            trainer = pl.Trainer(
                gpus=1,
                min_epochs=2,
                max_epochs=300,
                amp_level='O1',
                precision=16,
                
                limit_train_batches=300,
                limit_val_batches=30,
                logger=CSVLogger("../outputs", name=f'{dataset_name}_{model_name}'),
                callbacks=[
                    EarlyStopping(monitor='loss/val', patience=patience * 2, verbose=True),
                ],
            )

            # Train
            trainer.fit(model, dl_train, dl_val)

            ds_preds = predict(model.to(device),
                               ds_test,
                               batch_size * 2,
                               device=device,
                               scaler=dataset.output_scaler)

            print(dataset_name, model_name)
            print(f'mean_NLL {ds_preds.nll.mean().item():2.2f}')
            loss = ds_preds.nll.mean().item()

#             Performance TODO tensorboard, wandb
            print(plot_hist(trainer))
            plot_performance(ds_preds)

            metrics = dict(
                rmse=rmse(ds_preds.y_true, ds_preds.y_pred).item(), 
                smape=smape(ds_preds.y_true, ds_preds.y_pred).item(), 
                nll=ds_preds.nll.mean().item()
                )
            results[dataset_name][model_name] = metrics
            df_results = pd.concat({k:pd.DataFrame(v) for k,v in results.items()})
            display(df_results)
            
            dset_to_nc(ds_preds, Path(trainer.logger.experiment.log_dir)/'ds_preds.nc')
            model.cpu()
        except Exception as e:
            logging.exception('failed to run model')
            
df_results = pd.concat({k:pd.DataFrame(v) for k,v in results.items()})
display(df_results)


# -

# # Leaderboard

def bold_min(data):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'font-weight: bold'
    #remove % and cast to float
    data = data.replace('%','', regex=True).astype(float)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_min = data == data.min()
        return [attr if v else '' for v in is_min]
    else:  # from .apply(axis=None)
        is_min = data == data.min().min()
        return pd.DataFrame(np.where(is_min, attr, ''),
                            index=data.index, columns=data.columns)


print(f'Negative Log-Likelihood (NLL).\nover {window_future} steps')
d=df_results.xs('nll', level=1).T.round(2)
d.style.apply(bold_min)

print(f'Symmetric mean absolute percentage error (SMAPE)\nover {window_future} steps')
d=df_results.xs('smape', level=1).T.round(2)
d.style.apply(bold_min)
# # Plots
# # plots
# Load saved preds
results = defaultdict(dict)
for Dataset in datasets:
    dataset_name = Dataset.__name__
    for m_fn in models:
        pt_model = m_fn()
        model_name = type(pt_model).__name__

        checkpoint_name = f"{dataset_name}_{model_name}"
        save_dir = Path(f"../outputs")/checkpoint_name
        fs = sorted(save_dir.glob("**/ds_preds.nc"))
        if len(fs)>0:
            ds_preds = xr.open_dataset(fs[-1])
            results[dataset_name][model_name] = ds_preds

data_i = 100



# Plot mean of predictions
for dataset in results.keys():
    for model in results[dataset].keys():
        ds_preds = results[dataset][model]
        plot_prediction(ds_preds, data_i, label=f"{model}")
    plt.title(dataset)
    plt.legend()
    plt.show()

# +
dataset='BejingPM25'
n = len(results[dataset].keys())

plt.figure(figsize=(8, 1.5*n))
plt.suptitle(f'Plots with confidence for {dataset} ')
for i, model in enumerate(results[dataset].keys()):
    plt.subplot(n, 1, i+1)
    ds_preds = results[dataset][model]
    if i==n-1:
        # The last one has the legend
        plot_prediction(ds_preds, data_i, title=f"{model}", std=True, legend=True)
    else:
        plot_prediction(ds_preds, data_i, title=f"{model}", std=True, )
        
        # share the x axis
        locs, _ = plt.xticks()
        plt.xticks(locs, labels=[])
        plt.xlabel(None)
plt.subplots_adjust()
# -






