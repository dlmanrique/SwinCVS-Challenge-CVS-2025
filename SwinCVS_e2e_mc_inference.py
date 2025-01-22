##############################################################################################
##############################################################################################
print('Importing libraries...')
# Standard library imports
import os
import shutil
import time
import random
import json
from copy import copy, deepcopy
from pathlib import Path
from typing import Tuple, Dict
import warnings 

# Third-party imports
import torch
import torchvision
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm
from torch.utils.data import Sampler, Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import RandAugment
from torch.nn import Linear
from timm.data.auto_augment import auto_augment_transform

# Local imports
from scripts.functions_e2e import *

# IMPORTS FROM MSFT GITHUB
import yaml
from argparse import Namespace
from scripts.build import build_model, SwinLSTMModel
from types import SimpleNamespace

warnings.filterwarnings("ignore")

##############################################################################################
##############################################################################################
seed = 2
print(f"Current seed: {seed}")

# Environment Standardisation
random.seed(seed)                      # Set random seed
np.random.seed(seed)                   # Set NumPy seed
torch.manual_seed(seed)                # Set PyTorch seed
torch.cuda.manual_seed(seed)           # Set CUDA seed
torch.backends.cudnn.benchmark = False # Disable dynamic tuning
torch.use_deterministic_algorithms(True) # Force deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # CUDA workspace config

##############################################################################################
##############################################################################################
# Get path to current directory
pwd = Path.cwd()
print(f"cwd: {pwd}")

# Load config
cfg = 'config/SwinCVS.yaml'
config_dict = read_config(cfg)
config = config_to_yacs(config_dict)

# Load dataset dataframes
image_folder = pwd.parent / 'SurgLatentGraph/data/mmdet_datasets/endoscapes'
print(f"Dataset loaded from: {image_folder}")

train_dataframe, val_dataframe, test_dataframe = get_three_dataframes(image_folder, lstm=True)

##############################################################################################
##############################################################################################
# Endoscapes normalisation values
mean = [123.675/255, 116.28/255, 103.53/255]
std = [58.395/255, 57.12/255, 57.375/255]

# Change BGR to RGB
mean = mean[::-1]
std = std[::-1]

# Get model's image size
img_size = config.DATA.IMG_SIZE

# Create a transform sequence
transform_sequence = transforms.Compose([   transforms.CenterCrop(480),
                                            transforms.Resize((img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                mean=torch.tensor(mean),
                                                std=torch.tensor(std))
                                        ])

# Dataset
test_dataset = EndoscapesSwinLSTM_Dataset(test_dataframe[::5], transform_sequence)

# Dataloaders
test_dataloader = DataLoader(   test_dataset,
                                batch_size = 1,
                                shuffle = False,
                                pin_memory = True)

##############################################################################################
##############################################################################################
# Initialise SwinCVS according to config
model = None
model = build_model(config)
print('SwinCVS initialised successfully\n')

# Load saved weights for inference
model.load_state_dict(torch.load('weights/SwinV2LSTM_e2e_raw_mc_V3_sd5_bestMAP.pt'))
model.to('cuda')
print('Pretrained SwinCVS weights loaded successfully')

torch.cuda.empty_cache()

##############################################################################################
##############################################################################################
# Inference time measurement variables
start_time = 0
end_time = 0
times = []

# Performance measurement variables
test_probabilities = []
test_predictions = []
test_targets = []

len_dataloader = len(test_dataloader)

model.eval()
with torch.inference_mode():
    for idx, (samples, targets) in enumerate(test_dataloader):
        print(f"\rSample: {idx+1:04}/{len_dataloader:04}", end="")

        # Time start
        start_time = time.time()

        # Get preds
        samples, targets = samples.to('cuda'), targets.to('cuda')
        outputs_lstm = model(samples)

        # Get outputs
        test_probability = torch.sigmoid(outputs_lstm)
        test_prediction = torch.round(test_probability)

        # Time end
        end_time = time.time()
        elapsed_time  = end_time-start_time
        times.append(elapsed_time)

        # Save results from a batch to a list
        test_probabilities.append(test_probability.to('cpu'))
        test_predictions.append(test_prediction.to('cpu'))
        test_targets.append(targets.to('cpu'))

        torch.cuda.synchronize()

# Calculate metrics
C1_balanced_accuracy, C2_balanced_accuracy, C3_balanced_accuracy, total_balanced_accuracy = get_balanced_accuracies(test_targets, test_predictions)
C1_ap, C2_ap, C3_ap, mAP = get_map(test_targets, test_probabilities)

# Print metrics
print('\n\nTesting results:')
print('Average balanced accuracy', round((C1_balanced_accuracy+C1_balanced_accuracy+C3_balanced_accuracy)/3, 4))
print('C1 bacc', round(C1_balanced_accuracy, 4))
print('C2 bacc', round(C2_balanced_accuracy, 4))
print('C3 bacc', round(C3_balanced_accuracy, 4))
print('mAP', round(mAP, 4))
print('C1 ap', round(C1_ap, 4))
print('C2 ap', round(C2_ap, 4))
print('C3 ap', round(C3_ap, 4))

print(f"Inference time: mean={round(np.mean(times)*1000,1)}ms, std={round(np.std(times)*1000,1)}ms, total={round(np.sum(times),1)}s")