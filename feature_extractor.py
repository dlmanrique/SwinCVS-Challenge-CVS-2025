# This file extracts features and patch_embeddings from a pretrained SwinV2 model

import os
import json
import argparse
import torch
import warnings
import torch.nn as nn
from tqdm import tqdm

from scripts.f_build import build_model
from scripts.f_environment import get_config
from scripts.f_dataset import get_datasets, get_dataloaders
warnings.filterwarnings("ignore")

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description="Run SwinCVS with specified config")
parser.add_argument('--config_path', type=str, required=False, default='config/SwinCVS_config_feature_extract.yaml' , help='Path to config YAML file')
parser.add_argument('--fold', type=int, default=1, required=True)
parser.add_argument('--direction', type=str, required=False, default='None', choices=['past', 'both', 'future'])
parser.add_argument('--fps', type=int, required=False, default=0, choices=[10, 15, 30])
parser.add_argument('--extend_method', type=str, required=False, default='None', choices=['balanced', 'unbalanced'])
parser.add_argument('--frame_type_train', type=str, required=False, default='Original', choices=['Original', 'Preprocessed'])
parser.add_argument('--frame_type_test', type=str, required=False, default='Original', choices=['Original', 'Preprocessed'])
args = parser.parse_args()


config, experiment_name = get_config(args.config_path)

config.FOLD = args.fold
training_dataset, val_dataset, test_dataset = get_datasets(config, args)
train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config, training_dataset, val_dataset, test_dataset)

model = None
model = build_model(config)
print('Full model initialised successfully!\n')

best_model_fold = f'best_until_now/Fold{args.fold}/bestMAP.pt'
model.load_state_dict(torch.load(best_model_fold, weights_only=True))
print(f"Trained SwinCVS weights loaded successfully for INFERENCE - name: {best_model_fold}")
model.to('cuda')

model.eval()
model.head = nn.Identity()

# Create folder to save all this info


with torch.inference_mode():
    print('Calculating patch embbedings for val split')
    for idx, (samples, targets, image_path) in enumerate(tqdm(val_dataloader)):
        video_id = image_path[0].split('video_')[-1][:3]
        frame_id = image_path[0].split('/')[-1][:-4]

        samples, targets = samples.to('cuda'), targets.to('cuda')
        patch_embeddings = model.patch_embed(samples)
        last_layer_features = model(samples)
        
        # Delete batch dim
        patch_embeddings = patch_embeddings.squeeze().cpu()
        last_layer_features = last_layer_features.squeeze().cpu()

        #Create folders
        os.makedirs(f'features/patch_embeddings/video_{video_id}', exist_ok=True)
        os.makedirs(f'features/last_layer/video_{video_id}', exist_ok=True)

        #Save patch_embeds
        torch.save(patch_embeddings, f'features/patch_embeddings/video_{video_id}/{frame_id}.pt')

        #Save last layer features
        torch.save(last_layer_features, f'features/last_layer/video_{video_id}/{frame_id}.pt')

        torch.cuda.synchronize()


