from pathlib import Path
import torch

project_dir = Path(__file__).parent.parent

def to_numpy(x):
    """Helper function to avoid repeating code"""
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    return x
