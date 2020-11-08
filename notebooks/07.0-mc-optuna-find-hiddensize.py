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
# In this notebook we are going to find the optimal hidden_size for a model vs a dataset. We will use pytorch lightning and optuna.

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

from pathlib import Path
from tqdm.auto import tqdm

import pytorch_lightning as pl
# -
from seq2seq_time.data.dataset import Seq2SeqDataSet, Seq2SeqDataSets
from seq2seq_time.predict import predict, predict_multi
from seq2seq_time.util import dset_to_nc

# +
import logging
import warnings
import seq2seq_time.silence 
warnings.simplefilter('once')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', 'Consider increasing the value of the `num_workers` argument', UserWarning)
warnings.filterwarnings('ignore', 'Your val_dataloader has `shuffle=True`', UserWarning)

from pytorch_lightning import _logger as log
log.setLevel(logging.WARN)
# -

# ## Parameters

# +
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'using {device}')

timestamp = '20201108-095004'
print(timestamp)
window_past = 48*2
window_future = 48
batch_size = 64
num_workers = 5
datasets_root = Path('../data/processed/')
window_past
# -



# ## Datasets
#
# From easy to hard, these dataset show different challenges, all of them with more than 20k datapoints and with a regression output. See the 00.01 notebook for more details, and the code for more information.
#
# Some such as MetroInterstateTraffic are easier, some are periodic such as BejingPM25, some are conditional on inputs such as GasSensor, and some are noisy and periodic like IMOSCurrentsVel

from seq2seq_time.data.data import IMOSCurrentsVel, AppliancesEnergyPrediction, BejingPM25, GasSensor, MetroInterstateTraffic
datasets = [GasSensor, IMOSCurrentsVel, AppliancesEnergyPrediction, BejingPM25, MetroInterstateTraffic]
datasets
# ## Lightning
#
# We will use pytorch lightning to handle all the training scaffolding. We have a common pytorch lightning class that takes in the model and defines training steps and logging.

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
        return y_dist, extra

    def training_step(self, batch, batch_idx, phase='train'):
        x_past, y_past, x_future, y_future = batch
        y_dist, extra = self.forward(*batch)
        loss = -y_dist.log_prob(y_future).mean()
        self.log_dict({f'loss/{phase}':loss})
        if ('loss' in extra) and (phase=='train'):
            # some models have a special loss
            loss = extra['loss']
            self.log_dict({f'model_loss/{phase}':loss})
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, phase='val')
    
    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, phase='test')
    
    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr,  weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            patience=self.patience,
            verbose=False,
            min_lr=1e-7,
        ) if self.patience else None
        return {'optimizer': optim, 'lr_scheduler': scheduler, 'monitor': 'loss/val'}


# -

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger, TestTubeLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor


# ## Models

from seq2seq_time.models.baseline import BaselineLast, BaselineMean
from seq2seq_time.models.lstm_seq2seq import LSTMSeq2Seq
from seq2seq_time.models.lstm import LSTM
from seq2seq_time.models.transformer import Transformer
from seq2seq_time.models.transformer_seq2seq import TransformerSeq2Seq
from seq2seq_time.models.neural_process import RANP
from seq2seq_time.models.transformer_process import TransformerProcess
from seq2seq_time.models.tcn import TCNSeq
from seq2seq_time.models.inceptiontime import InceptionTimeSeq
from seq2seq_time.models.xattention import CrossAttention
# +
import gc

def free_mem():
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
# -



# +
# PARAMS: model
dropout=0.0
layers=6
nhead=4

models = [
#     lambda xs, ys: BaselineLast(),
#     lambda xs, ys, hidden_size: BaselineMean(),
    lambda xs, ys, hidden_size, layers: Transformer(xs,
                ys,
                attention_dropout=dropout,
                nhead=nhead,
                nlayers=layers,
                hidden_size=hidden_size),

    lambda xs, ys, hidden_size, layers:TransformerProcess(xs,
                ys, hidden_size=hidden_size, nhead=nhead,
        latent_dim=hidden_size//2, dropout=dropout,
        nlayers=layers),
    lambda xs, ys, hidden_size, layers:TCNSeq(xs, ys, hidden_size=hidden_size, nlayers=layers, dropout=dropout, kernel_size=2),
    lambda xs, ys, hidden_size, layers: RANP(xs,
        ys, hidden_dim=hidden_size, dropout=dropout, 
         latent_dim=hidden_size//2, n_decoder_layers=layers, n_latent_encoder_layers=layers, n_det_encoder_layers=layers),
    lambda xs, ys, hidden_size, layers: TransformerSeq2Seq(xs,
                       ys,
                       hidden_size=hidden_size,
                       nhead=nhead,
                       nlayers=layers,
                       attention_dropout=dropout
                                     ),
    lambda xs, ys, hidden_size, layers: LSTM(xs,
         ys,
         hidden_size=hidden_size,
         lstm_layers=layers//2,
         lstm_dropout=dropout),
    lambda xs, ys, hidden_size, layers: LSTMSeq2Seq(xs,
                ys,
                hidden_size=hidden_size,
                lstm_layers=layers//2,
                lstm_dropout=dropout),
    lambda xs, ys, hidden_size, layers: CrossAttention(xs,
                ys,
                nlayers=layers,
                hidden_size=hidden_size,),
    lambda xs, ys, hidden_size, layers: InceptionTimeSeq(xs,
                ys,
                kernel_size=96,
                layers=layers//2,
                hidden_size=hidden_size,
                bottleneck=hidden_size//4)

]
# +
# DEBUG: sanity check

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
        free_mem()
        pt_model = m_fn(xs, ys, 8, 4)
        model_name = type(pt_model).__name__
        print(timestamp, dataset_name, model_name)

        # Wrap in lightning
        model = PL_MODEL(pt_model,
                         lr=3e-4
                        ).to(device)
        trainer = pl.Trainer(
            fast_dev_run=True,
            # GPU
            gpus=1,
            amp_level='O1',
            precision=16,
        )
# -

# Lets summarize all models, and make sure they have a similar number of parameters

# ## Train

from collections import defaultdict
from seq2seq_time.metrics import rmse, smape


max_iters=20000


tensorboard_dir = Path(f"../outputs/{timestamp}").resolve()
print(f'For tensorboard run:\ntensorboard --logdir="{tensorboard_dir}"')



# +
class MetricsCallback(pl.Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

def objective(trial):
    # sample
    hidden_size_exp = trial.suggest_int("hidden_size_exp", 2, 8)
    hidden_size = 2**hidden_size_exp
    
    layers = trial.suggest_int("layers", 2, 12)
    
    # Load model
    pt_model = m_fn(xs, ys, hidden_size, layers)
    model_name = type(pt_model).__name__
    
    # Wrap in lightning
    patience = 2
    model = PL_MODEL(pt_model,
                     lr=3e-4, patience=patience,
                    weight_decay=4e-5
                    ).to(device)

    
    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We don't use any logger here as it requires us to implement several abstract
    # methods. Instead we setup a simple callback, that saves metrics from each validation step.
#     metrics_callback = MetricsCallback()
    
    save_dir = f"../outputs/{timestamp}/{dataset_name}_{model_name}/{trial.number}"
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    trainer = pl.Trainer(
        # Training length
        min_epochs=2,
        max_epochs=100,
        limit_train_batches=max_iters//batch_size,
        limit_val_batches=max_iters//batch_size//5,
        # Misc
        gradient_clip_val=20,
        terminate_on_nan=True,
        # GPU
        gpus=1,
        amp_level='O1',
        precision=16,
        # Callbacks
        default_root_dir=save_dir,
        logger=False,
        callbacks=[
#             metrics_callback, 
                   EarlyStopping(monitor='loss/val', patience=patience * 2),
                   PyTorchLightningPruningCallback(trial, monitor="loss/val")],
    )
    trainer.fit(model, dl_train, dl_val)
    
    # Run on all val data, using test mode
    r = trainer.test(model, test_dataloader=dl_val, verbose=False)
    return r[0]['loss/test']
# -



import optuna
from optuna.integration import PyTorchLightningPruningCallback

Path(f"../outputs/{timestamp}").mkdir(exist_ok=True)
results = defaultdict(dict)
for Dataset in tqdm(datasets, desc='datasets'):
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
                         shuffle=False,
                         batch_size=batch_size,
                         num_workers=num_workers)

    for i, m_fn in enumerate(tqdm(models, desc=f'models ({dataset_name})')):
        try:
            model_name = type(m_fn(8, 8, 8, 2)).__name__
            free_mem()
            study_name = f'{timestamp}_{dataset_name}-{model_name}'
            
            storage = f"sqlite:///../outputs/{timestamp}/optuna.db"
            pruner = optuna.pruners.MedianPruner()
            study = optuna.create_study(storage=storage, 
                                        study_name=study_name, 
                                        pruner=pruner,
                                        load_if_exists=True)
            study.optimize(objective, n_trials=100, timeout=60*60)
            print("Number of finished trials: {}".format(len(study.trials)))

            print("Best trial:")
            trial = study.best_trial

            print("  Value: {}".format(trial.value))

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
            
        except Exception as e:
            logging.exception('failed to run model')


