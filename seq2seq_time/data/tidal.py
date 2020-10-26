import uptide
import pandas as pd
import numpy as np

# https://en.wikipedia.org/wiki/Theory_of_tides#Harmonic_analysis
default_tidal_constituents = [
    'M2',
    'S2',
    'N2',
    'K2',  # Semi-diurnal
    'K1',
    'O1',
    'P1',
    'Q1',  # Diurnal
    'M4',
    'M6',
    'S4',
    'MK3',  # Short period
    'MM',
    'SSA',
    'SA'  # Long period
]


def generate_tidal_periods(t: pd.Series,
                           constituents: list = default_tidal_constituents):
    tide = uptide.Tides(constituents)
    t0 = t[0]
    td = t - t0
    td = td.dt.total_seconds().to_numpy().astype(int)
    tide.set_initial_time(t0)

    # calc tides
    amplitudes = np.ones_like(td)
    phases = np.zeros_like(td)
    eta = {}
    for name, f, amplitude, omega, phase, phi, u in zip(
            tide.constituents, tide.f, amplitudes, tide.omega, phases,
            tide.phi, tide.u):
        eta[name] = f * amplitude * np.cos(omega * td - phase + phi + u)
    df_eta = pd.DataFrame(eta, index=t)
    return df_eta


