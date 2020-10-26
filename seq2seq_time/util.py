from pathlib import Path
import torch
import xarray as xr
import logging

logger = logging.getLogger(__file__)
project_dir = Path(__file__).parent.parent

def to_numpy(x):
    """Helper function to avoid repeating code"""
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    return x

def mask_upper_triangular(N, device):
    """Causal attention."""
    return torch.triu(torch.ones(N, N), diagonal=1).to(device).bool()

def dset_to_nc(dset, f, engine="netcdf4", compression={"zlib": True}):
    if isinstance(dset, xr.DataArray):
        dset = dset.to_dataset(name="data")
    encoding = {k: {"zlib": True} for k in dset.data_vars}
    logger.info(f"saving to {f}")
    dset.to_netcdf(f, engine=engine, encoding=encoding)
    logger.info(f"Wrote {f.stem}.nc size={f.stat().st_size/1e6} M")
