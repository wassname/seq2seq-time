import sklearn
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn_pandas import DataFrameMapper

def normalize_encode_dataframe(df, encoder=OrdinalEncoder):
    """Normalise numeric data, encode categorical data."""
    columns_input_numeric = list(df._get_numeric_data().columns)
    columns_categorical = list(set(df.columns)-set(columns_input_numeric))
    
    transformers= [([n], StandardScaler()) for n in columns_input_numeric] + \
                  [([n], encoder()) for n in columns_categorical]
    scaler = DataFrameMapper(transformers, df_out=True)
    df_norm = scaler.fit_transform(df)
    return df_norm, scaler
    
def timeseries_split(df, test_fraction=0.2, dropna=None):
    """Split timeseries data with test in the future"""
    
    # If there are lots of nan's we can ignore them when splitting into portions
    if isinstance(dropna, list):
        index = df.dropna(subset=dropna).index
    elif dropna is True:
        index = df.dropna().index
    else:
        index = df.index
    
    i = int(len(index)*(1- test_fraction))
    dt = index.values[i]
    return df.loc[:dt], df.loc[dt:]
