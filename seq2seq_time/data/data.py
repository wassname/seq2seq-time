from typing import List, Tuple
from torchvision.datasets.utils import download_url, extract_archive, download_and_extract_archive
import os
from tqdm.auto import tqdm
from pathlib import Path
from sklearn_pandas import DataFrameMapper
import xarray as xr
import pandas as pd
import numpy as np
import zipfile

from .dataset import Seq2SeqDataSet
from .util import normalize_encode_dataframe, timeseries_split
from ..util import dset_to_nc
from .tidal import generate_tidal_periods


class RegressionForecastData:   
    columns_forecast = None # The input colums which can be included in future (e.g. week or weather forecast)
    columns_target = None # Target columns
    
    def __init__(self, datasets_root):        
        self.datasets_root = datasets_root
        
        # Process data
        self.df = self.download()        
        self.df_norm, self.scaler = self.normalize(self.df)
        self.output_scaler = next(filter(lambda r:r[0][0] in self.columns_target, self.scaler.features))[-1]
        self.df_train, self.df_val, self.df_test = self.split(self.df_norm)
        
        # Check processing
        self.check()

    @property
    def columns_past(self):
        return set(self.df.columns)-set(self.columns_forecast)-set(self.columns_target)
    
    def download(self) -> pd.DataFrame:
        """Implement this method to download data and return raw df"""
        raise NotImplementedError()
        return df
    
    def normalize(self, df) -> Tuple[pd.DataFrame, DataFrameMapper]:
        df_norm, scaler = normalize_encode_dataframe(df)
        return df_norm, scaler
    
    def split(self, df_norm: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_train, df_test = timeseries_split(df_norm)
        df_test, df_val  = timeseries_split(df_test, 0.5)
        return df_train, df_val, df_test
    
    def check(self) -> None:
        """Check the resulting dataframe"""
        assert isinstance(self.df.index, pd.DatetimeIndex), 'index must be datetime'
        assert self.df.index.freq is not None, 'df must have freq'        
        assert self.columns_forecast is not None
        assert self.columns_target is not None
        assert ~set(self.columns_target).issubset(set(self.columns_forecast)), 'target columns should not be in forecast'
        assert set(self.columns_forecast).issubset(set(self.df.columns)), 'columns_forecast must be in df'
        assert set(self.columns_target).issubset(set(self.df.columns)), 'columns_target must be in df'
        
    def to_datasets(self, window_past: int, window_future: int, valid:bool=False) -> Tuple[Seq2SeqDataSet, Seq2SeqDataSet]:
        """Convert to torch datasets"""
        ds_train = Seq2SeqDataSet(self.df_train, window_past=window_past, window_future=window_future, columns_target=self.columns_target, columns_past=self.columns_past)
        ds_val = Seq2SeqDataSet(self.df_val, window_past=window_past, window_future=window_future, columns_target=self.columns_target, columns_past=self.columns_past)
        ds_test = Seq2SeqDataSet(self.df_test, window_past=window_past, window_future=window_future, columns_target=self.columns_target, columns_past=self.columns_past)
        return ds_train, ds_val, ds_test
    
    def __repr__(self):
        return f'<{type(self).__name__} {self.df.shape if (self.df is not None) else None}>'

class GasSensor(RegressionForecastData):
    """
    See: http://archive.ics.uci.edu/ml/datasets/Gas+sensor+array+temperature+modulation
    """
    
    columns_target = ['R1 (MOhm)']
    columns_forecast = ['Flow rate (mL/min)', 'Heater voltage (V)']
    
    def download(self):
        # TODO cache in faster format
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00487/gas-sensor-array-temperature-modulation.zip'
        
        # download if needed
        # extract_path = self.datasets_root/'gas-sensor-array-temperature-modulation.zip'
        download_url(url, self.datasets_root)
        outfile = self.datasets_root / 'gas-sensor-array-temperature-modulation.pk'
        if not outfile.exists():
        
            # Load csv's from inside zip
            zf = zipfile.ZipFile(self.datasets_root / 'gas-sensor-array-temperature-modulation.zip')
            dfs=[]
            for f in zf.namelist():
                if f.endswith('.csv'):
                    now = pd.to_datetime(Path(f).stem, format='%Y%m%d_%H%M%S')
                    df = pd.read_csv(zf.open(f))
                    df.index = pd.to_timedelta(df['Time (s)'], unit='s') + now
                    dfs.append(df)
            self.df = pd.concat(dfs).dropna(subset=self.columns_target)

            df = df[[ 'CO (ppm)', 'Humidity (%r.h.)', 'Temperature (C)',
                'Flow rate (mL/min)', 'Heater voltage (V)', 'R1 (MOhm)']]
            df = df.resample('0.3S').first()

            df.to_pickle(outfile)
        df = pd.read_pickle(outfile)
        return df


class MetroInterstateTraffic(RegressionForecastData):
    """
    See: https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume
    """
    
    columns_target = ['traffic_volume']
    columns_forecast = ['holiday', 'month', 'day', 'week', 'hour',
       'minute', 'dayofweek']
    
    def download(self):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz'
        
        # download if needed
        filename = '00492_Metro_Interstate_Traffic_Volume.csv.gz'
        local_path = self.datasets_root/filename
        if not local_path.exists():
            download_url(url, self.datasets_root, filename)
        df = (pd.read_csv(local_path, index_col='date_time', parse_dates=['date_time'])
              .dropna(subset=self.columns_target)
              .resample('1H').first()
             )
        
        # Make holiday a bool
        df['holiday'] = ~df['holiday'].isna()
        df['weather_main'] = df['weather_main'].fillna('none')
        df['weather_description'] = df['weather_description'].fillna('none')
        
        # Add time features 
        time = df.index.to_series()
        df["month"] = time.dt.month
        df['day'] = time.dt.day
        df['week'] = time.dt.isocalendar().week
        df['hour'] = time.dt.hour
        df['minute'] = time.dt.minute
        df['dayofweek'] = time.dt.dayofweek
        
        return df

class AppliancesEnergyPrediction(RegressionForecastData):
    """
    See: https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
    """
    
    columns_target = ['log_Appliances']
    columns_forecast = ['month', 'day', 'week', 'hour',
       'minute', 'dayofweek']
    
    def download(self):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv'
        
        # download if needed
        filename = '00374_AppliancesEnergyPrediction.csv'
        local_path = self.datasets_root/filename
        if not local_path.exists():
            download_url(url, self.datasets_root, filename)
        df = pd.read_csv(local_path, index_col='date', parse_dates=['date'])
        
        # log target
        df['log_Appliances'] = np.log(df['Appliances'] + 1e-5)
        df = df.drop(columns=['Appliances'])
        df = df.dropna(subset=self.columns_target).resample('10T').first()
        
        # Add time features 
        time = df.index.to_series()
        df["month"] = time.dt.month
        df['day'] = time.dt.day
        df['week'] = time.dt.isocalendar().week
        df['hour'] = time.dt.hour
        df['minute'] = time.dt.minute
        df['dayofweek'] = time.dt.dayofweek
        
        return df

class BejingPM25(RegressionForecastData):
    """
    See: http://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
    """
    
    columns_target = ['log_pm2.5']
    columns_forecast = ['month', 'day', 'week', 'hour',
       'minute', 'dayofweek']
    
    def download(self):
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv'
        
        # download if needed
        filename = '00381_BejingPM25.csv'
        local_path = self.datasets_root/filename
        if not local_path.exists():
            download_url(url, self.datasets_root, filename)
        df = pd.read_csv(local_path)
        df.index = pd.to_datetime(df[['year', 'month', 'day', 'hour']]).dt.tz_localize('Asia/Shanghai')
        df = df.drop(columns=['year', 'month', 'day', 'hour', 'No'])
        
        # log target
        df['log_pm2.5'] = np.log(df['pm2.5'] + 1e-5)
        df = df.drop(columns=['pm2.5'])
        
        df.dropna(subset=self.columns_target, inplace=True)
        df = df.resample('1H').first()
        
        df['cbwd'] = df['cbwd'].fillna('none')
        
        
        
        # Add time features 
        time = df.index.to_series()
        df["month"] = time.dt.month
        df['day'] = time.dt.day
        df['week'] = time.dt.isocalendar().week
        df['hour'] = time.dt.hour
        df['minute'] = time.dt.minute
        df['dayofweek'] = time.dt.dayofweek
        
#         df['log_pm2.5'] = np.log(df['pm2.5']+1e-5)
        
        return df

def get_current_timeseries(
        cache_folder=Path("../data/raw/IMOS_ANMN/"),
        outfile=Path(
            '../data/processed/currents/MOS_ANMN-WA_AETVZ_WATR20_FV01_WATR20-1909-Continental-194_currents.nc'
        )):
    """
    Download Current data from the IMOS and pre-process.
    """
    if not outfile.exists():

        files = [
            "IMOS_ANMN-WA_AETVZ_20090715T080000Z_WATR20_FV01_WATR20-0907-Continental-194_END-20090716T181317Z_C-20191122T052830Z.nc",
            "IMOS_ANMN-WA_AETVZ_20100409T080000Z_WATR20_FV01_WATR20-1004-Continental-194_END-20100430T084500Z_C-20191122T053845Z.nc",
            "IMOS_ANMN-WA_AETVZ_20101222T080000Z_WATR20_FV01_WATR20-1012-Continental-194_END-20110518T051500Z_C-20200916T020035Z.nc",
            "IMOS_ANMN-WA_AETVZ_20110608T080000Z_WATR20_FV01_WATR20-1106-Continental-194_END-20111122T035000Z_C-20200916T025619Z.nc",
            "IMOS_ANMN-WA_AETVZ_20111221T060300Z_WATR20_FV01_WATR20-1112-Continental-194_END-20120704T050500Z_C-20200916T043212Z.nc",
            "IMOS_ANMN-WA_AETVZ_20120726T044000Z_WATR20_FV01_WATR20-1207-Continental-194_END-20130204T044000Z_C-20200916T032027Z.nc",
            "IMOS_ANMN-WA_AETVZ_20130221T080000Z_WATR20_FV01_WATR20-1302-Continental-194_END-20131003T035000Z_C-20180529T020609Z.nc",
            "IMOS_ANMN-WA_AETVZ_20131111T080000Z_WATR20_FV01_WATR20-1311-Continental-194_END-20140519T035000Z_C-20200114T033335Z.nc",
            "IMOS_ANMN-WA_AETVZ_20140710T080000Z_WATR20_FV01_WATR20-1407-Continental-194_END-20150121T021500Z_C-20180529T055902Z.nc",
            "IMOS_ANMN-WA_AETVZ_20150213T080000Z_WATR20_FV01_WATR20-1502-Continental-194_END-20150424T134002Z_C-20200114T035347Z.nc",
            "IMOS_ANMN-WA_AETVZ_20150914T080000Z_WATR20_FV01_WATR20-1509-Continental-194_END-20160331T043000Z_C-20180601T013623Z.nc",
            "IMOS_ANMN-WA_AETVZ_20160427T080000Z_WATR20_FV01_WATR20-1604-Continental-194_END-20160531T021800Z_C-20180531T071709Z.nc",
            #     "IMOS_ANMN-WA_AETVZ_20170512T080000Z_WATR20_FV01_WATR20-1705-Continental-194_END-20170717T014558Z_C-20190805T004647Z.nc",
            "IMOS_ANMN-WA_AETVZ_20171204T080000Z_WATR20_FV01_WATR20-1712-Continental-194_END-20180618T030000Z_C-20180620T233149Z.nc",
            "IMOS_ANMN-WA_AETVZ_20180802T080000Z_WATR20_FV01_WATR20-1807-Continental-194_END-20190225T054500Z_C-20190227T001343Z.nc",
            "IMOS_ANMN-WA_AETVZ_20190307T080000Z_WATR20_FV01_WATR20-1903-Continental-194_END-20190911T003144Z_C-20200114T045053Z.nc",
            "IMOS_ANMN-WA_AETVZ_20190926T080000Z_WATR20_FV01_WATR20-1909-Continental-194_END-20200326T030000Z_C-20200420T064334Z.nc",
        ]
        base = "http://thredds.aodn.org.au/thredds/fileServer/IMOS/ANMN/WA/WATR20/Velocity/"

        # Download files
        [download_url(base + f, cache_folder) for f in files]

        # load and merge
        xds = [xr.open_dataset(cache_folder / f) for f in files]
        vars = [
            'VCUR', 'UCUR', 'WCUR', 'TEMP', 'PRES_REL', 'DEPTH', 'ROLL',
            'PITCH'
        ]
        xds2 = [x[vars].isel(HEIGHT_ABOVE_SENSOR=18) for x in xds]
        xd = xr.concat(xds2, dim='TIME')
        xd = xd.where(xd.DEPTH > 150)  # remove outliers

        xd['TIME'] = xd['TIME'].dt.round('10T')
        xd = xd.dropna(dim='TIME', subset=['VCUR', 'UCUR', 'WCUR'])

        # Generate tidal freqs
        t = xd.TIME.to_series()
        df_eta = generate_tidal_periods(t)

        # Add tidal freqs
        xd = xd.merge(df_eta)

        dset_to_nc(xd, outfile)
    return outfile


class IMOSCurrentsVel(RegressionForecastData):
    """
    
    Current Speed at ANMN Two Rocks, WA, 204m mooring
    
    see:
    - http://thredds.aodn.org.au/thredds/fileServer/IMOS/ANMN/WA/WATR20/Velocity/
    from https://catalogue-imos.aodn.org.au/geonetwork/srv/api/records/ae86e2f5-eaaf-459e-a405-e654d85adb9c
    and http://thredds.aodn.org.au/thredds/catalog/IMOS/ANMN/WA/WATR20/Velocity/catalog.html
    And https://en.wikipedia.org/wiki/Theory_of_tides
    """

    columns_target = ['SPD']
    columns_forecast = [
        'M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1', 'M4', 'M6', 'S4',
        'MK3', 'MM', 'SSA', 'SA'
    ]

    def download(self):
        outfile = self.datasets_root / 'MOS_ANMN-WA_AETVZ_WATR20_FV01_WATR20-1909-Continental-194_currents.nc'
        get_current_timeseries(outfile=outfile)

        # made in previous notebook
        xd = xr.load_dataset(outfile)
        df = xd.to_dataframe().drop(
            columns=['HEIGHT_ABOVE_SENSOR', 'NOMINAL_DEPTH'])
        df['SPD'] = np.sqrt(df.VCUR**2 + df.UCUR**2)
        df.dropna(subset=self.columns_target, inplace=True)
        df = df.resample('30T').first()[:'2015']

        return df
