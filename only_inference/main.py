# This is the main.py file for this inference
print('Importing libraries...')
# Standard library imports
import argparse
import os
import random

# Third-party imports
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

# Local imports
from scripts.f_environment import get_config, set_deterministic_behaviour
from scripts.f_dataset import get_datasets, get_dataloaders
from scripts.f_build import build_model
from evaluator import get_map


# Load configuration
parser = argparse.ArgumentParser(description="Run SwinCVS with specified config")
parser.add_argument('--config_path', type=str, required=False, default='config/SwinCVS_config.yaml' , help='Path to config YAML file')
parser.add_argument('--ckpt_path',  type=str, required=False)
args = parser.parse_args()

config = get_config(args.config_path)

seed = config.SEED
# Environment Standardisation
random.seed(seed)                      # Set random seed
np.random.seed(seed)                   # Set NumPy seed
torch.manual_seed(seed)                # Set PyTorch seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(seed)           # Set CUDA seed
torch.use_deterministic_algorithms(True) # Force deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # CUDA workspace config



##############################################################################################
##############################################################################################
# DATASET and DATALOADER
test_dataset = get_datasets(config)
test_dataloader = get_dataloaders(config, test_dataset)

##############################################################################################
##############################################################################################

# Initialise SwinCVS according to config
model = None
model = build_model(config)
model.head = nn.Linear(in_features=1024, out_features=3, bias=True)
print('Full model initialised successfully!\n')

# Load saved weights for inference
model.load_state_dict(torch.load(args.ckpt_path, weights_only=True))
print(f"Trained SwinCVS weights loaded successfully for INFERENCE - name: {args.ckpt_path}")
model.to('cuda')
torch.cuda.empty_cache()


print('\nTesting')
model.eval()
# Performance measurement variables
test_probabilities = []
test_predictions = []
test_targets = []

with torch.inference_mode():
    for idx, (samples, targets) in enumerate(tqdm(test_dataloader)):

        # Get preds
        samples, targets = samples.to('cuda'), targets.to('cuda')
        
        if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER and not config.MODEL.INFERENCE:
            outputs_swin, outputs_lstm = model(samples)
        else:
            outputs_lstm = model(samples)

        # Get outputs
        test_probability = torch.sigmoid(outputs_lstm)
        test_prediction = torch.round(test_probability)


        # Save results from a batch to a list
        test_probabilities.append(test_probability.to('cpu'))
        test_predictions.append(test_prediction.to('cpu'))
        test_targets.append(targets.to('cpu'))

        torch.cuda.synchronize()

# Calculate metrics
C1_ap, C2_ap, C3_ap, mAP = get_map(test_targets, test_probabilities)

# Print metrics
print('\nTesting results:')
print('mAP', round(mAP, 4))
print('C1 ap', round(C1_ap, 4))
print('C2 ap', round(C2_ap, 4))
print('C3 ap', round(C3_ap, 4))
