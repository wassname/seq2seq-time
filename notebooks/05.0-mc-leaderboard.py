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

import warnings
warnings.simplefilter('once')

from seq2seq_time.data.dataset import Seq2SeqDataSet, Seq2SeqDataSets
from seq2seq_time.predict import predict, predict_multi
from seq2seq_time.util import dset_to_nc

import logging, sys
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import datashade, dynspread
hv.extension('bokeh')


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


# -

# ## Plot helpers

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
        output_size, hidden_dim=32, 
         latent_dim=64, n_decoder_layers=4),
    lambda: LSTM(input_size,
         output_size,
         hidden_size=80,
         lstm_layers=3,
         lstm_dropout=0.3),
    lambda: LSTMSeq2Seq(input_size,
                output_size,
                hidden_size=64,
                lstm_layers=2,
                lstm_dropout=0.25),
    lambda: TransformerSeq2Seq(input_size,
                       output_size,
                       hidden_size=128,
                       nhead=8,
                       nlayers=4,
                       attention_dropout=0.2),
    lambda: Transformer(input_size,
                output_size,
                attention_dropout=0.2,
                nhead=8,
                nlayers=8,
                hidden_size=128),
    lambda :TransformerProcess(input_size,
                output_size, hidden_size=16,
        latent_dim=8, dropout=0.5,
        nlayers=4,)
#     lambda :TemporalConvNet()
]
# models

# +
from seq2seq_time.data.data import IMOSCurrentsVel, AppliancesEnergyPrediction, BejingPM25, GasSensor, MetroInterstateTraffic

datasets = [IMOSCurrentsVel, BejingPM25, GasSensor, AppliancesEnergyPrediction, MetroInterstateTraffic]
datasets

# +
# GasSensor(datasets_root)
# -

# ## Train

from collections import defaultdict
results = defaultdict(dict)

from seq2seq_time.metrics import rmse, smape

for Dataset in datasets:
    dataset_name = Dataset.__name__
    dataset = Dataset(datasets_root)
    ds_train, ds_test = dataset.to_datasets(window_past=window_past,
                                            window_future=window_future)

    # Init data
    x_past, y_past, x_future, y_future = ds_train.get_rows(10)
    input_size = x_past.shape[-1]
    output_size = y_future.shape[-1]


# +
for Dataset in datasets:
    dataset_name = Dataset.__name__
    dataset = Dataset(datasets_root)
    ds_train, ds_test = dataset.to_datasets(window_past=window_past,
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
    dl_test = DataLoader(ds_test,
                         batch_size=batch_size,
                         num_workers=num_workers)

    for m_fn in models:
        try:
            free_mem()
            pt_model = m_fn()
            model_name = type(pt_model).__name__
            print(dataset_name, model_name)

            # Wrap in lightning
            patience = 2
            model = PL_MODEL(pt_model,
                             lr=3e-4, patience=patience,
                             weight_decay=1e-5).to(device)

            # Trainer            
            trainer = pl.Trainer(
                gpus=1,
                min_epochs=2,
                max_epochs=20,
                amp_level='O1',
                precision=16,
                
                limit_train_batches=1000,
                limit_val_batches=100,
                logger=CSVLogger("../outputs", name=f'{dataset_name}_{model_name}'),
                callbacks=[
                    EarlyStopping(monitor='loss/val', patience=patience * 2, verbose=True),
                ],
            )

            # Train
            trainer.fit(model, dl_train, dl_test)

            ds_preds = predict(model.to(device),
                               ds_test,
                               batch_size * 2,
                               device=device,
                               scaler=dataset.output_scaler)

            print(dataset_name, model_name)
            print(f'mean_NLL {ds_preds.nll.mean().item():2.2f}')
            loss = ds_preds.nll.mean().item()

            # Performance
#             print(plot_hist(trainer))
#             plot_performance(ds_preds)

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

# +
#   File "/media/wassname/Storage5/projects2/3ST/seq2seq-time/seq2seq_time/models/transformer.py", line 54, in forward
#     outputs = self.encoder(x, mask=mask#, src_key_padding_mask=x_key_padding_mask
#                              File "/media/wassname/Storage5/projects2/3ST/seq2seq-time/seq2seq_time/models/transformer.py", line 54, in forward
#     outputs = self.encoder(x, mask=mask#, src_key_padding_mask=x_key_padding_mask
# -

df_results.xs('nll', level=1).round(2)

# +
# ds_preds.to_netcdf(trainer.logger.experiment.log_dir+'/ds_preds2.nc')
# -



# # Plots

