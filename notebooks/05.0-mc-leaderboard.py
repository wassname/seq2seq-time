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
# - [ ] make overlap between past and future

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

import warnings
warnings.simplefilter('once')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

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

from pathlib import Path
from tqdm.auto import tqdm

import pytorch_lightning as pl
# +
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import datashade, dynspread
hv.extension('bokeh', inline=True)
from seq2seq_time.visualization.hv_ggplot import ggplot_theme
hv.renderer('bokeh').theme = ggplot_theme

# holoview datashader timeseries options
# %opts RGB [width=800 height=200 show_grid=True active_tools=["xwheel_zoom"] default_tools=["xpan","xwheel_zoom", "reset", "hover"] toolbar="right"]
# %opts Curve [width=800 height=200 show_grid=True active_tools=["xwheel_zoom"] default_tools=["xpan","xwheel_zoom", "reset", "hover"] toolbar="right"]
# %opts Scatter [width=800 height=200 show_grid=True active_tools=["xwheel_zoom"] default_tools=["xpan","xwheel_zoom", "reset", "hover"] toolbar="right"]
# %opts Layout [width=800 height=200]
# -


from seq2seq_time.data.dataset import Seq2SeqDataSet, Seq2SeqDataSets
from seq2seq_time.predict import predict, predict_multi
from seq2seq_time.util import dset_to_nc

# ## Parameters

# +
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'using {device}')

window_past = 48*2
window_future = 48*2
batch_size = 128
num_workers = 4
datasets_root = Path('../data/processed/')
window_past


# -

# ## Plot helpers

# +
def hv_plot_std(d: xr.Dataset):
    xf = d.t_target
    yp = d.y_pred
    s = d.y_pred_std
    return hv.Spread((xf, yp, s * 2),
                     label='2*std').opts(alpha=0.5, line_width=0)

def hv_plot_pred(d: xr.Dataset):
    # Get arrays
    xf = d.t_target
    yp = d.y_pred
    s = d.y_pred_std
    return hv.Curve({'x': xf, 'y': yp})

def hv_plot_true(d: xr.Dataset):
    """Plot a prediction into the future, at a single point in time.""" 
    
     # Plot true
    x = np.concatenate([d.t_past, d.t_target])
    yt = np.concatenate([d.y_past, d.y_true])
    p = hv.Scatter({
        'x': x,
        'y': yt
    }, label='true').opts(color='black')


    
    now=pd.Timestamp(d.t_source.squeeze().values)
        
    p = p.opts(
        ylabel=str(ds_preds.attrs['targets']),
        xlabel=f'{now}'
    )

    
    # plot a red line for now
    p *= hv.VLine(now, label='now').opts(color='red', framewise=True)

    return p

def hv_plot_prediction(d):
    p = hv_plot_true(d)
    p *= hv_plot_pred(d)
    p *= hv_plot_std(d)
    return p


# +
def plot_performance(ds_preds, full=False):
    """Multiple plots using xr_preds"""
    p = hv_plot_prediction(ds_preds.isel(t_source=10))
    display(p)

    n = len(ds_preds.t_source)
    d_ahead = ds_preds.mean(['t_source'])['nll'].groupby('t_ahead_hours').mean()
    nll_vs_tahead = (hv.Curve(
        (d_ahead.t_ahead_hours,
         d_ahead)).redim(x='hours ahead',
                         y='nll').opts(
                                       title=f'NLL vs time ahead (no. samples={n})'))
    display(nll_vs_tahead)

    # Make a plot of the NLL over time. Does this solution get worse with time?
    if full:
        d_source = ds_preds.mean(['t_ahead'])['nll'].groupby('t_source').mean()
        nll_vs_time = (hv.Curve(d_source).opts(
                                               title='Error vs time of prediction'))
        display(nll_vs_time)

    # A scatter plot is easy with xarray
    if full:
        tlim = (ds_preds.y_true.min().item(), ds_preds.y_true.max().item())
        true_vs_pred = datashade(hv.Scatter(
            (ds_preds.y_true,
             ds_preds.y_pred))).redim(x='true', y='pred').opts(width=400,
                                                               height=400,
                                                               xlim=tlim,
                                                               ylim=tlim,
                                                               title='Scatter plot')
        true_vs_pred = dynspread(true_vs_pred)
        true_vs_pred
        display(true_vs_pred)

def plot_hist(trainer):
    try:
        df_hist = pd.read_csv(trainer.logger.experiment.metrics_file_path)
        df_hist['epoch'] = df_hist['epoch'].ffill()
            
        df_histe = df_hist.set_index('epoch').groupby('epoch').mean()
        if len(df_histe)>1:
            p = hv.Curve(df_histe, kdims=['epoch'], vdims=['loss/train']).relabel('train')
            p *= hv.Curve(df_histe, kdims=['epoch'], vdims=['loss/val']).relabel('val')
            display(p.opts(ylabel='loss'))
        return df_histe
    except Exception as e:
        print(e)
        pass


# +
def df_bold_min(data):
    '''
    highlight the maximum in a Series or DataFrame
    
    
    Usage:
        `df.style.apply(df_bold_min)`
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
    
def format_results(results, metric=None):
    df_results = pd.concat({k:pd.DataFrame(v) for k,v in results.items()}).T
    if metric:
        return df_results.xs(metric, axis=1, level=1).rename_axis(columns=metric)
    return df_results

def display_results(results, metric='nll', strformat="{:.2f}"):
    df_results = format_results(results, metric=metric)
    
    # display metric
    display(df_results
            .style.format(strformat)
            .apply(df_bold_min)
           )
# -



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
    display(p)

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


# # Models

# +

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
from seq2seq_time.models.tcn import TCNSeq2Seq
# ## Plots


# +
import gc

def free_mem():
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


# +
hidden_size = 32
dropout=0.25
layers=6
nhead=8

models = [
    lambda xs, ys: BaselineLast(),
#     lambda xs, ys: TransformerAutoR(xs,
#         ys, hidden_out_size=hidden_size),
    lambda xs, ys: RANP(xs,
        ys, hidden_dim=hidden_size, dropout=dropout, 
         latent_dim=hidden_size//4, n_decoder_layers=layers),
#     lambda xs, ys: LSTM(xs,
#          ys,
#          hidden_size=hidden_size,
#          lstm_layers=layers,
#          lstm_dropout=dropout),
#     lambda xs, ys: LSTMSeq2Seq(xs,
#                 ys,
#                 hidden_size=hidden_size,
#                 lstm_layers=layers,
#                 lstm_dropout=dropout),
#     lambda xs, ys: TransformerSeq2Seq(xs,
#                        ys,
#                        hidden_size=hidden_size,
#                        nhead=nhead,
#                        nlayers=layers,
#                        attention_dropout=dropout),
    lambda xs, ys: Transformer(xs,
                ys,
                attention_dropout=dropout,
                nhead=nhead,
                nlayers=layers,
                hidden_size=hidden_size),
#     lambda xs, ys:TransformerProcess(xs,
#                 ys, hidden_size=hidden_size,
#         latent_dim=hidden_size//4, dropout=dropout,
#         nlayers=layers,)
    lambda xs, ys:TCNSeq2Seq(xs, ys, hidden_size=hidden_size, nlayers=layers, dropout=dropout)
]
# models
# -



# +
# Summarize each models shape and weights
Dataset = datasets[0]
dataset = Dataset(datasets_root)
ds_train, ds_val, ds_test = dataset.to_datasets(window_past=window_past,
                                        window_future=window_future)
dl_val = DataLoader(ds_val, batch_size=batch_size)
x_past, y_past, x_future, y_future = next(iter(dl_val))
xs = x_past.shape[-1]
ys = y_future.shape[-1]

from seq2seq_time.torchsummaryX import summary
sizes=[]
for m_fn in models:
    pt_model = m_fn(xs, ys)
    model_name = type(pt_model).__name__
    with torch.no_grad():
        df_summary, df_total = summary(pt_model, x_past, y_past, x_future, y_future, print_summary=False)
    sizes.append(df_total.rename(columns={'Totals':model_name}))
df_model_sizes = pd.concat(sizes, 1)
df_model_sizes.style.format(pd.io.formats.format.EngFormatter(use_eng_prefix=True))
# -
# ## Train

from collections import defaultdict
results = defaultdict(dict)




from seq2seq_time.metrics import rmse, smape

# +
for Dataset in datasets:
    dataset_name = Dataset.__name__
    dataset = Dataset(datasets_root)
    ds_train, ds_val, ds_test = dataset.to_datasets(window_past=window_past,
                                            window_future=window_future)

    # Init data
    x_past, y_past, x_future, y_future = ds_train.get_rows(10)
    xs = x_past.shape[-1]
    ys = y_future.shape[-1]

    # Loaders
    dl_train = DataLoader(ds_train,
                          batch_size=batch_size,
                          shuffle=True,
                          pin_memory=num_workers == 0,
                          num_workers=num_workers)
    dl_val = DataLoader(ds_val,
                         shuffle=True,
                         batch_size=batch_size,
                         num_workers=num_workers)

    for m_fn in models:
        try:
            free_mem()
            pt_model = m_fn(xs, ys)
            model_name = type(pt_model).__name__
            print(dataset_name, model_name)

            # Wrap in lightning
            patience = 5
            model = PL_MODEL(pt_model,
                             lr=3e-4, patience=patience,
#                              weight_decay=4e-5
                            ).to(device)

            # Trainer            
            trainer = pl.Trainer(
                gpus=1,
                min_epochs=2,
                max_epochs=300,
                amp_level='O1',
                precision=16,
                
                limit_train_batches=800,
                limit_val_batches=150,
                logger=CSVLogger("../outputs", name=f'{dataset_name}_{model_name}'),
                callbacks=[
                    EarlyStopping(monitor='loss/val', patience=patience * 2),
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
            display_results(results, 'nll')
            
            dset_to_nc(ds_preds, Path(trainer.logger.experiment.log_dir)/'ds_preds.nc')
            model.cpu()
        except Exception as e:
            logging.exception('failed to run model')
            
df_results = pd.concat({k:pd.DataFrame(v) for k,v in results.items()})
display(df_results)
# -
# # Leaderboard

print(f'Negative Log-Likelihood (NLL).\nover {window_future} steps')
df_results = pd.concat({k:pd.DataFrame(v) for k,v in results.items()})
display_results(results, 'nll')

# # Plots

# +

# Load saved preds
ds_predss = defaultdict(dict)
for Dataset in datasets:
    dataset_name = Dataset.__name__
    for m_fn in models:
        pt_model = m_fn(xs, ys)
        model_name = type(pt_model).__name__

        checkpoint_name = f"{dataset_name}_{model_name}"
        save_dir = Path(f"../outputs")/checkpoint_name
        fs = sorted(save_dir.glob("**/ds_preds.nc"))
        if len(fs)>0:
            ds_preds = xr.open_dataset(fs[-1])
            ds_predss[dataset_name][model_name] = ds_preds
# -

data_i = 100

# Plot mean of predictions
n = hv.Layout()
for dataset in ds_predss.keys():
    d = next(iter(ds_predss[dataset].values())).isel(t_source=data_i)
    p = hv_plot_true(d)
    for model in results[dataset].keys():
        ds_preds = ds_predss[dataset][model]
        d = ds_preds.isel(t_source=data_i)
        p *= hv_plot_pred(d).relabel(label=f"{model}")
    n += p.opts(title=dataset, legend_position='top_left')
n.cols(1).opts(shared_axes=False)

dataset='BejingPM25'
n = hv.Layout()
for i, model in enumerate(ds_predss[dataset].keys()):
    ds_preds = ds_predss[dataset][model]
    d = ds_preds.isel(t_source=data_i)
    p = hv_plot_true(d)
    p *= hv_plot_pred(d).relabel('pred')
    p *= hv_plot_std(d)
    n += p.opts(title=f'{dataset} {model}', legend_position='top_left')
n.cols(1)

plot_performance(ds_preds, full=True)












