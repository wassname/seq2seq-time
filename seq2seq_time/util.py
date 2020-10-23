from pathlib import Path
import torch

project_dir = Path(__file__).parent.parent

def to_numpy(x):
    """Helper function to avoid repeating code"""
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    return x

def mask_upper_triangular(N, device):
    """Causal attention."""
    return torch.triu(torch.ones(N, N), diagonal=1).to(device).bool()
