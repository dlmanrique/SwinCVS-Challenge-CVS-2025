import torch
import os
from pathlib import Path

def save_weights(model, checkpoint_path, epoch):
    """
    Function saving weights of passed model.
    """
    
    print(f"weights path: {checkpoint_path}")

    checkpoint_dir = os.path.join(checkpoint_path, f'bestMAP_epoch_{epoch}.pt')
    torch.save(model.state_dict(), checkpoint_dir)