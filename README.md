seq2seq-time
==============================

Using sequence to sequence (and normal) interfaces for multivariate timeseries regression.

Since this is a deep learning approach it's hard to do hyperparameter optimisation for every model, so be aware that these are only indicative number. The most interesting results are which models are robust, and which models fail for certain dataset.

NOTE: This is a work in progress, with out final numbers...

<img src="reports/figures/Seq2Seq for regression.png" />


To run some code start with [notebooks/05.5-mc-leaderboard.ipynb](notebooks/05.5-mc-leaderboard.ipynb)

# Results


## Negative Log Likelihood

After trying 20+ differen't hidden sizes and layer combinations, here are the best values:

| model              |   AppliancesEnergyPred |   BejingPM25 |   GasSensor |   IMOSCurrentsVel |   MetroInterstateTraffic |
|:-------------------|-----------------------------:|-------------:|------------:|------------------:|-------------------------:|
| BaselineLast       |                         1.48 |         1.55 |        1.97 |              0.89 |                     1.74 |
| BaselineMean       |                         1.32 |         1.44 |        1.58 |              1.2  |                     1.41 |
| CrossAttention     |                         1.55 |         1.41 |       -0.64 |              1.66 |                    -0.1  |
| InceptionTimeSeq   |                         1.1  |         1.24 |       -2.1  |              0.85 |                    -0.16 |
| LSTM               |                         1.17 |         1.27 |       -1.54 |              0.88 |                    -0.2  |
| LSTMSeq2Seq        |                         1.2  |         1.29 |       -1.49 |              0.89 |                    -0.2  |
| RANP               |                         1.28 |         1.43 |       -2.13 |              1.04 |                    -0.29 |
| TCNSeq             |                         1.08 |         1.24 |       -1.74 |              0.82 |                    -0.32 |
| Transformer        |                         1.2  |         1.3  |       -1.96 |              0.88 |                    -0.25 |
| TransformerProcess |                         1.16 |         1.4  |       -0.88 |              1.39 |                    -0.3  |
| TransformerSeq2Seq |                         1.17 |         2.39 |        0.34 |              1.27 |                    -0.19 |

RANP is a Recurrent attentive neural process. Implementation details and hyperparameters can be found by reading the code starting with [notebooks/07.1-mc-optuna.ipynb](notebooks/07.1-mc-optuna.ipynb)

If we scale it so baseline last is 0, and the best performance is -1, we can compare all datasets (lower is better)

mean of scaled performance over all datasets
| model              |     0 |
|:-------------------|------:|
| TCNSeq             | -0.98 |
| InceptionTimeSeq   | -0.89 |
| LSTM               | -0.72 |
| Transformer        | -0.7  |
| LSTMSeq2Seq        | -0.65 |
| RANP               | -0.13 |
| BaselineLast       |  0    |
| BaselineMean       |  0.73 |
| TransformerProcess |  0.91 |
| TransformerSeq2Seq |  1.25 |
| CrossAttention     |  1.91 |

## Datasets

To ensure a robust score we use multiple multivariate regression timeseries.

For more see [notebooks/01.0-mc-datasets.ipynb](notebooks/01.0-mc-datasets.ipynb) or [notebooks/01.0-mc-datasets/index.html](notebooks/01.0-mc-datasets/index.html)

![](reports/figures/data_batches_appliances.png)

Applience energy usage prediction.

![](reports/figures/data_batches_currents.png)

30 minute, current speed at Two Rocks 200m Mooring. Has tidal periods as extra features.

![](reports/figures/data_batches_gas.png)

A metal oxide (MOX) gas sensor exposed during 3 weeks to mixtures of carbon monoxide and humid synthetic air in a gas chamber.

![](reports/figures/data_batches_pm25.png)

Hourly PM2.5 data of US Embassy in Beijing. This measures smoke as well as some pollen, fog, and dust particles of a certain size. Weather data from a nearby airport are included.

![](reports/figures/data_batches_traffic.png)

Hourly Minneapolis-St Paul, MN traffic volume for westbound I-94. Includes weather and holiday features from 2012-2018.

## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploratio    │
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements       <- The requirements folder for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── seq2seq_time       <- Source code for use in this project.
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

```python

```
