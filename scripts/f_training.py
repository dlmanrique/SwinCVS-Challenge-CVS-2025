import torch
import os
from pathlib import Path

def save_weights(model, config, experiment_name, mAP, best_MAP, epoch):
    if mAP >= best_MAP:
        print(f"New best result (Epoch {epoch+1}), saving weights...")
        best_MAP = mAP
        if config.TRAIN.CHECKPOINT_PATH == None:
            checkpoint_path = Path.cwd() / 'weights'
        else:
            checkpoint_path = config.TRAIN.CHECKPOINT_PATH
        checkpoint_dir = os.path.join(checkpoint_path, f'{experiment_name}_bestMAP.pt')
        torch.save(model.state_dict(), checkpoint_dir)
    else:
        print('\n')