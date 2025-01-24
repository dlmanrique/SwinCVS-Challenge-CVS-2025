import yaml
from yacs.config import CfgNode as CN
import random
import numpy as np
import torch
import os
import re

def get_config(config_path):
    """
    Runs functions related to reading, processing, and informing user about the config setup.
    """
    config_dict = read_config(config_path)
    config = config_to_yacs(config_dict)
    experiment_name = validate_config(config)
    return config, experiment_name

def read_config(config_file):
    """
    Read Yaml file into dict
    """
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def config_to_yacs(dict_config):
    """
    Convert the dict to a yacs.
    """
    if not isinstance(dict_config, dict):
        return dict_config  # Return non-dict values as is
    cfg = CN()
    for key, value in dict_config.items():
        cfg[key] = config_to_yacs(value)  # Recursively convert nested dictionaries
    return cfg

def validate_config(config):
    """
    Analyse config, inform user on the chosen settings, set an experiment model name. 
    """
    print(f"\nModel settings:")
    # Check model setup settings
    if config.MODEL.LSTM and config.MODEL.E2E and config.MODEL.MULTICLASSIFIER:
        print("End-to-end SwinCVS: SwinV2 backbone, with LSTM and 'multiclassifier'")
        experiment_name = 'SwinCVS_E2E_MC'
    elif config.MODEL.LSTM and config.MODEL.E2E and not config.MODEL.MULTICLASSIFIER:
        print("End-to-end SwinCVS without multiclassifier")
        experiment_name = 'SwinCVS_E2E'
    elif config.MODEL.LSTM and not config.MODEL.E2E:
        print("Frozen SwinCVS: frozen weights in SwinV2 backbone, with LSTM classifier")
        experiment_name = 'SwinCVS_frozen'
    elif not config.MODEL.LSTM:
        print("SwinV2 model selected (not SwinCVS!)")
        experiment_name = 'SwinV2_backbone'
    else:
        print("Custom model settings detected...")
        print(f"LSTM={config.MODEL.LSTM} | E2E={config.MODEL.E2E} | MULTICLASSIFIER={config.MODEL.MULTICLASSIFIER}")
        experiment_name = f"CustomModel"
    # Check if inference
    if config.MODEL.INFERENCE:
        assert config.MODEL.INFERENCE_WEIGHTS is not None, "Model selected for INFERENCE, but no weights were provided!"
        print("Script set for inference - NOT TRAINING.")

    # Otherwise confirm backbone weights
    else:
        try:
            if 'swinv2_base_patch4' in config.BACKBONE.PRETRAINED:
                print("Backbone weights: ImageNet")
                experiment_name += "_IMNP"
            elif config.BACKBONE.PRETRAINED is not None:
                print(f"Backbone weights: '{config.BACKBONE.PRETRAINED}'")
                experiment_name += "_ENDP"
        except:
            print('Backbone weights: None!')
    
    if config.MODEL.INFERENCE:
        if 'sd' in config.MODEL.INFERENCE_WEIGHTS:
            inference_seed = find_seed_in_weight(config.MODEL.INFERENCE_WEIGHTS)
            experiment_name += f"_sd{inference_seed}"
            experiment_name += '_INFERENCE'
        else:
            experiment_name += f"_sd{config.SEED}"

        return experiment_name
    
    experiment_name += f"_sd{config.SEED}"

    return experiment_name

def find_seed_in_weight(weight_name):
    match = re.search(r'sd(\d+)', weight_name)
    if match:
        return match.group(1)
    else:
        return False
    
def set_deterministic_behaviour(seed):
    # Environment Standardisation
    random.seed(seed)                      # Set random seed
    np.random.seed(seed)                   # Set NumPy seed
    torch.manual_seed(seed)                # Set PyTorch seed
    torch.cuda.manual_seed(seed)           # Set CUDA seed
    torch.backends.cudnn.benchmark = False # Disable dynamic tuning
    torch.use_deterministic_algorithms(True) # Force deterministic behavior
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # CUDA workspace config