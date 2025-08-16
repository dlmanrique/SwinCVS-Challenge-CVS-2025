# This is the main.py file for this inference
print('Importing libraries...')
# Standard library imports
import argparse
import time
import json
import os

# Third-party imports
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

# Local imports
from scripts.f_environment import get_config, set_deterministic_behaviour
from scripts.f_dataset import get_datasets, get_dataloaders
from scripts.f_build import build_model


# Load configuration
parser = argparse.ArgumentParser(description="Run SwinCVS with specified config")
parser.add_argument('--config_path', type=str, required=False, default='config/SwinCVS_config.yaml' , help='Path to config YAML file')
args = parser.parse_args()

config = get_config(args.config_path)

# Seed all procces
seed = config.SEED
set_deterministic_behaviour(seed)

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
print('Full model initialised successfully!\n')

# Load saved weights for inference
if config.MODEL.INFERENCE:
    weights = "weights/" + config.MODEL.INFERENCE_WEIGHTS
    model.load_state_dict(torch.load(weights))
    print(f"Trained SwinCVS weights loaded successfully for INFERENCE - name: {config.MODEL.INFERENCE_WEIGHTS}")
model.to('cuda')
torch.cuda.empty_cache()


print('\nTesting')
model.eval()
with torch.inference_mode():
    for idx, (samples, targets) in enumerate(tqdm(test_dataloader)):

        # Time start
        start_time = time.time()

        # Get preds
        samples, targets = samples.to('cuda'), targets.to('cuda')
        
        if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER and not config.MODEL.INFERENCE:
            outputs_swin, outputs_lstm = model(samples)
        else:
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
print('\nTesting results:')
print('Average balanced accuracy', round((C1_balanced_accuracy+C1_balanced_accuracy+C3_balanced_accuracy)/3, 4))
print('C1 bacc', round(C1_balanced_accuracy, 4))
print('C2 bacc', round(C2_balanced_accuracy, 4))
print('C3 bacc', round(C3_balanced_accuracy, 4))
print('mAP', round(mAP, 4))
print('C1 ap', round(C1_ap, 4))
print('C2 ap', round(C2_ap, 4))
print('C3 ap', round(C3_ap, 4))

mean_inference = round(np.mean(times)*1000,1)
std_inference = round(np.std(times)*1000,1)
total_inference = round(np.sum(times),1)
print(f"Inference time: mean={mean_inference}ms, std={std_inference}ms, total={total_inference}s")

test_predictions_2save = torch.cat(test_predictions, dim=0).tolist()
test_probabilities_2save = torch.cat(test_probabilities, dim=0).tolist()
test_targets_2save = torch.cat(test_targets, dim=0).tolist()

epoch_results = {'avg_bal_acc': round(total_balanced_accuracy, 4),
                'C1_bacc': round(C1_balanced_accuracy, 4), 'C2_bacc': round(C2_balanced_accuracy, 4), 'C3_bacc': round(C3_balanced_accuracy, 4),
                'avg_map': round(mAP, 4),
                'C1_map': round(C1_ap, 4), 'C2_map': round(C2_ap, 4), 'C3_map': round(C3_ap, 4),
                'preds': test_predictions_2save, 'true': test_targets_2save, 'preds_prob': test_probabilities_2save,
                'mean_inference_ms': mean_inference, 'std_inference_ms': std_inference, 'total_inference_s': total_inference}
results_dict[f"Testing_@E{best_epoch+1}"] = epoch_results # CHANGE THE BEST EPOCH

results_file_path = '/'.join(checkpoint_path.split('/')[:-1])
with open(os.path.join(results_file_path, f'results.json'), 'w') as file:
    json.dump(results_dict, file, indent=4)