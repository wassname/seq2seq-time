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
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm.auto import tqdm
from IPython.display import display, HTML
# -
import warnings
warnings.simplefilter('once')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# +
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import datashade, dynspread
hv.extension('bokeh', inline=True)
from seq2seq_time.visualization.hv_ggplot import ggplot_theme
hv.renderer('bokeh').theme = ggplot_theme
hv.archive.auto()

# holoview datashader timeseries options
# %opts RGB [width=800 height=200 show_grid=True active_tools=["xwheel_zoom"] default_tools=["xpan","xwheel_zoom", "reset", "hover"] toolbar="right"]
# %opts Curve [width=800 height=200 show_grid=True active_tools=["xwheel_zoom"] default_tools=["xpan","xwheel_zoom", "reset", "hover"] toolbar="right"]
# %opts Scatter [width=800 height=200 show_grid=True active_tools=["xwheel_zoom"] default_tools=["xpan","xwheel_zoom", "reset", "hover"] toolbar="right"]
# %opts Layout [width=800 height=200]
# -


# ## Parameters

# +
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f'using {device}')

window_past = 48*2
window_future = 48
batch_size = 4
datasets_root = Path('../data/processed/')
# -

# ## Plot helpers

# ## Datasets

# +
from seq2seq_time.data.data import IMOSCurrentsVel, AppliancesEnergyPrediction, BejingPM25, GasSensor, MetroInterstateTraffic

datasets = [IMOSCurrentsVel, BejingPM25, GasSensor, AppliancesEnergyPrediction, MetroInterstateTraffic, ]
datasets
# +
# plot a batch
def plot_batch_y(ds, i):
    x_past, y_past, x_future, y_future = ds.get_rows(i)
    y = pd.concat([y_past, y_future])
    p = hv.Scatter(y)

    now = y_past.index[-1]
    p *= hv.VLine(now).relabel('now').opts(color='red')
    return p

def plot_batches_y(dataset, window_past=window_past, window_future=window_future):
    ds_name = type(dataset).__name__
    opts=dict(width=200, height=100, xaxis=None, yaxis=None)
    ds_train, ds_val, ds_test = d.to_datasets(window_past=window_past,
                                              window_future=window_future)
    n = 4
    max_i = min(len(ds_train), len(ds_val), len(ds_test))
    ii = list(np.linspace(0, max_i-10, n-1).astype(int)) + [-1]
    l = hv.Layout()
    for i in ii:
        l += plot_batch_y(ds_train, i).opts(title=f'train {i}', **opts) 
        l += plot_batch_y(ds_val, i).opts(title=f'val {i}', **opts)
        l += plot_batch_y(ds_test, i).opts(title=f'test {i}', **opts)
    return l.opts(shared_axes=False, toolbar='right', title=f"{ds_name} freq={d.df.index.freq.freqstr}").cols(3)


# -

for dataset in datasets:
    d = dataset(datasets_root)
    display(HTML(f"<h3>{dataset.__name__}</h3>"))
    print(d.__doc__)
    print(f'{len(d)} rows at freq{d.df.index.freq.freqstr}')
    print('columns_forecast', d.columns_forecast)
    print('columns_past', d.columns_past)
    print('columns_target', d.columns_target)
    print
    display(d.df)



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
    p = p.opts(title=f"{dataset.__name__}, n={len(d)}, freq={d.df.index.freq.freqstr}")
    display(p)



# View train, test, val splits
for dataset in datasets:
    ds_name = type(dataset).__name__
    d = dataset(datasets_root)
    print(d)
    display(plot_batches_y(d))

# +
# def plot_batch_x(ds, i):
#     """Plot input features"""
#     x_past, y_past, x_future, y_future = ds.get_rows(i)
#     x = pd.concat([x_past, x_future])
#     p = hv.NdOverlay({
#         col: hv.Curve(x[col]) for col in x.columns
#     }, kdims='column')
#     now = y_past.index[-1]
#     p *= hv.VLine(now).relabel('now').opts(color='red')
#     return p

# def plot_batches_x(d):
#     """Plot input features for multiple batch"""
#     ds_train, ds_val, ds_test = d.to_datasets(window_past=window_past,
#                                               window_future=window_future)
#     l = plot_batch_x(ds_train, 10) + plot_batch_x(ds_val, 10) + plot_batch_x(ds_test, 10)
#     l = l.cols(1).opts(shared_axes=False, title=f'{type(d).__name__}')
#     return l

# +
# ds_train, ds_val, ds_test = d.to_datasets(window_past=window_past,
#                                           window_future=window_future)

# +
# # View input columns
# for dataset in datasets:
#     d = dataset(datasets_root)
#     display(plot_batches_x(d))
# -

hv.archive.export()
hv.archive.last_export_status()




