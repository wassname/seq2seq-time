import xarray as xr 
import torch
from tqdm.auto import tqdm
import pandas as pd

from .util import to_numpy

def predict(model, ds_test, batch_size, device='cpu', scaler=None):
    """
    Gather all predictions into xarray.
    
    When we generate prediction in a sequence to sequence model we start at a time then predict
    N steps into the future. So we have 2 dimensions: source time, target time.

    But we also care about how far we were predicting into the future, so we have 3 dimensions: source time, target time, time ahead.

    It's hard to use pandas for data with virtual dimensions so we will use xarray. Xarray has an interface similar to pandas but also allows coordinates which are virtual dimensions.
    """
    load_test = torch.utils.data.dataloader.DataLoader(ds_test, batch_size=batch_size)
    freq = ds_test.t.freq
    xrs = []
    for i, batch in enumerate(tqdm(load_test, desc='predict')):
        model.eval()
        with torch.no_grad():
            x_past, y_past, x_future, y_future = [d.to(device) for d in batch]
            y_dist = model(x_past, y_past, x_future, y_future)
            nll = -y_dist.log_prob(y_future)

            # Convert to numpy
            mean = to_numpy(y_dist.loc.squeeze(-1))
            std = to_numpy(y_dist.scale.squeeze(-1))
            nll = to_numpy(nll.squeeze(-1))
            y_future = to_numpy(y_future.squeeze(-1))
            y_past = to_numpy(y_past.squeeze(-1))    

        # Make an xarray.Dataset for the data
        bs = y_future.shape[0]
        t_source = ds_test.t[i:i+bs].values
        t_ahead = pd.timedelta_range(0, periods=ds_test.window_future, freq=freq).values
        t_behind = pd.timedelta_range(end=-pd.Timedelta(freq), periods=ds_test.window_past, freq=freq)
        xr_out = xr.Dataset(
            {
                # Format> name: ([dimensions,...], array),
                "y_past": (["t_source", "t_behind",], y_past),
                "nll": (["t_source", "t_ahead",], nll),
                "y_pred": (["t_source", "t_ahead",], mean),
                "y_pred_std": (["t_source", "t_ahead",], std),
                "y_true": (["t_source", "t_ahead",], y_future),
            },
            coords={"t_source": t_source, "t_ahead": t_ahead, "t_behind": t_behind},
        )
        xrs.append(xr_out)

    # Join all batches
    ds_preds = xr.concat(xrs, dim="t_source")
    
    # undo scaling on y
    if scaler:
        ds_preds['y_pred_std'].values = ds_preds.y_pred_std * scaler.scale_
        ds_preds['y_past'].values =  scaler.inverse_transform(ds_preds.y_past)
        ds_preds['y_pred'].values =  scaler.inverse_transform(ds_preds.y_pred)
        ds_preds['y_true'].values =  scaler.inverse_transform(ds_preds.y_true)

    # Add some derived coordinates, they will be the ones not in bold
    # The target time, is a function of the source time, and how far we predict ahead
    ds_preds = ds_preds.assign_coords(t_target=ds_preds.t_source+ds_preds.t_ahead)

    ds_preds = ds_preds.assign_coords(t_past=ds_preds.t_source+ds_preds.t_behind)

    # Some plots don't like timedeltas, so lets make a coordinate for time ahead in hours
    ds_preds = ds_preds.assign_coords(t_ahead_hours=(ds_preds.t_ahead*1.0e-9/60/60).astype(float))
    return ds_preds
