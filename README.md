seq2seq-time
==============================

Using sequence to sequence (and normal) interfaces for multivariate timeseries regression.

Since this is a deep learning approach it's hard to do hyperparameter optimisation for every model, so be aware that these are only indicative number. The most interesting results are which models are robust, and which models fail for certain dataset.

NOTE: This is a work in progress, with out final numbers...

<img src="reports/figures/Seq2Seq for regression.png" />


  
# Results


NOTE: Draft numbers

- [ ] TODO mean over N runs
- [ ] TODO hyperparameter opt to make sure I'm comparing optimal hidden_size

See [notebooks/05.5-mc-leaderboard.ipynb](notebooks/05.5-mc-leaderboard.ipynb)

## Negative Log Likelihood

|                    |   GasSensor |   IMOSCurrentsVel |   AppliancesEnergyPrediction |   BejingPM25 |   MetroInterstateTraffic |   mean(e-e_baseline) |
|:-------------------|------------:|------------------:|-----------------------------:|-------------:|-------------------------:|---------------------:|
| RANP               |       -1.91 |              0.93 |                         1.25 |         1.39 |                    -0.36 |                -1.16 |
| TransformerProcess |       -0.84 |              1.02 |                         1.17 |         1.43 |                    -0.33 |                -0.93 |
| Transformer        |       -1.18 |              0.93 |                         1.8  |         1.31 |                    -0.37 |                -0.92 |
| TCNSeq             |       -0.47 |              0.88 |                         1.1  |         1.28 |                    -0.15 |                -0.89 |
| CrossAttention     |       -0.58 |              1.27 |                         1.24 |         1.45 |                    -0.34 |                -0.81 |
| LSTMSeq2Seq        |        0    |              0.95 |                         1.2  |         1.28 |                    -0.29 |                -0.79 |
| LSTM               |       -0.2  |              0.97 |                         1.34 |         1.29 |                    -0.05 |                -0.75 |
| TransformerSeq2Seq |        0.69 |              1.49 |                         1.54 |         1.49 |                    -0.31 |                -0.43 |
| InceptionTimeSeq   |       -2.07 |              1.31 |                         4.65 |         1.32 |                    -0.03 |                -0.38 |
| BaselineMean       |        1.54 |              1.1  |                         1.41 |         1.59 |                     1.43 |                 0    |


## Model sizes

|                    | Total params   | Trainable params   |   Non-trainable params | Mult-Adds   |
|:-------------------|:---------------|:-------------------|-----------------------:|:------------|
| BaselineMean       | 1.0            | 1.0                |                      0 | 0.0         |
| Transformer        | 32.562k        | 32.562k            |                      0 | 31.088k     |
| TransformerProcess | 72.722k        | 72.722k            |                      0 | 101.088k    |
| TCNSeq             | 6.258k         | 6.258k             |                      0 | 1.84272M    |
| RANP               | 21.626k        | 21.626k            |                      0 | 24.256k     |
| TransformerSeq2Seq | 71.794k        | 71.794k            |                      0 | 68.368k     |
| LSTM               | 6.05k          | 6.05k              |                      0 | 5.664k      |
| LSTMSeq2Seq        | 12.002k        | 12.002k            |                      0 | 11.232k     |
| CrossAttention     | 44.642k        | 44.642k            |                      0 | 42.64k      |
| InceptionTimeSeq   | 46.346k        | 46.346k            |                      0 | 6.543744M   |

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
