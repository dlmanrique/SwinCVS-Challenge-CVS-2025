import torch
import os

def save_weights(model, config, experiment_name, mAP, best_MAP, epoch):
    if mAP >= best_MAP:
        print(f"New best result (Epoch {epoch+1}), saving weights...")
        best_MAP = mAP
        checkpoint_dir = os.path.join(config.TRAIN.CHECKPOINT_PATH, f'{experiment_name}_bestMAP.pt')
        torch.save(model.state_dict(), checkpoint_dir)
    else:
        print('\n')