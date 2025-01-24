# Standard library imports
import os
import shutil
import random
import json
from copy import copy, deepcopy
from pathlib import Path
from typing import Tuple, Dict
import multiprocessing as mp

# Third-party imports
import torch
import torchvision
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm
from torch.utils.data import Sampler, Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from math import inf
from timm.scheduler.cosine_lr import CosineLRScheduler
import yaml
from yacs.config import CfgNode as CN
import re

# FUNCTIONS
def find_seed_in_weight(weight_name):
    match = re.search(r'sd(\d+)', weight_name)
    if match:
        return match.group(1)
    else:
        return False

def update_params(alpha, beta, epoch):
    if epoch <= 4:
        pass
    else:
        alpha -= 0.04
        beta += 0.04
        print(f"New alpha/beta = {alpha, beta} @ epoch {epoch+1}")
    return alpha, beta

def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)

    lr_scheduler = None
    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=(num_steps - warmup_steps) if config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX else num_steps,
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX
        )
    return lr_scheduler

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def build_optimizer(config, model, **kwargs):
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    parameters, no_decay_names = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None

    # SwinCVS with Multiclassifier (requires E2E=True)
    if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER:
            parameters2= [  {'params': model.swinv2_model.parameters(), 'lr': config.TRAIN.OPTIMIZER.ENCODER_LR},
                            {'params': model.fc_swin.parameters(), 'lr': config.TRAIN.OPTIMIZER.ENCODER_LR},
                            {'params': model.lstm.parameters(), 'lr': config.TRAIN.OPTIMIZER.CLASSIFIER_LR},
                            {'params': model.fc_lstm.parameters(), 'lr': config.TRAIN.OPTIMIZER.CLASSIFIER_LR}]
    # SwinCVS without Multiclassifier
    elif config.MODEL.LSTM:
        parameters2= [  {'params': model.swinv2_model.parameters(), 'lr': config.TRAIN.OPTIMIZER.ENCODER_LR},
                        {'params': model.lstm.parameters(), 'lr': config.TRAIN.OPTIMIZER.CLASSIFIER_LR },
                        {'params': model.fc_lstm.parameters(), 'lr': config.TRAIN.OPTIMIZER.CLASSIFIER_LR}]
    # Bare backbone - swinV2
    else:
        parameters2= [  {'params': model.parameters(), 'lr': config.TRAIN.OPTIMIZER.ENCODER_LR}]

    if opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters2, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    else:
        raise NotImplementedError

    return optimizer

def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    no_decay_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_names.append(name)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    

    param_groups = [{'params': has_decay}, {'params': no_decay, 'weight_decay': 0.}]

    return param_groups, no_decay_names 

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

class CVSSampler(Sampler):
    def __init__(self, dataset, upsample_factor, reshuffle=True):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        self.labels = [tuple(label.item() for label in dataset[idx][1]) for idx in range(len(dataset))]  # Convert labels to tuples
        self.upsample_factor = upsample_factor
        self.reshuffle = reshuffle
        self.upsampled_indices = self.create_upsampled_indices()

    def create_upsampled_indices(self):
        upsampled_indices = []
        label_map = {
            (0., 0., 0.): 0,
            (1., 0., 0.): 1,
            (0., 1., 0.): 2,
            (0., 0., 1.): 3,
            (1., 1., 0.): 4,
            (1., 0., 1.): 5,
            (0., 1., 1.): 6,
            (1., 1., 1.): 7
        }
        for idx, label in zip(self.indices, self.labels):
            label_tuple = tuple(label)
            if label_tuple in label_map:
                scale = self.upsample_factor[label_map[label_tuple]]
                upsampled_indices.extend([idx] * int(scale))
                if random.random() < scale - int(scale):
                    upsampled_indices.append(idx)
            else:
                raise RuntimeError(f"Data label outside of expected value. Label {label_tuple}")
        return upsampled_indices

    def __iter__(self):
        if self.reshuffle:
            random.shuffle(self.upsampled_indices)
        return iter(self.upsampled_indices)

    def __len__(self):
        return len(self.upsampled_indices)
    

def get_balanced_accuracies(y_true, y_pred):
    true_labels = np.concatenate([np.array(x) for x in y_true])
    predicted_labels = np.concatenate([np.array(x) for x in y_pred])

    C1_true = deepcopy(true_labels)
    C1_predicted = deepcopy(predicted_labels)
    C1_true = np.delete(C1_true, [1,2], axis=1)
    C1_predicted = np.delete(C1_predicted, [1,2], axis=1)

    C2_true = deepcopy(true_labels)
    C2_true = np.delete(C2_true, [0,2], axis=1)
    C2_predicted = deepcopy(predicted_labels)
    C2_predicted = np.delete(C2_predicted, [0,2], axis=1)

    C3_true = deepcopy(true_labels)
    C3_predicted = deepcopy(predicted_labels)
    C3_true = np.delete(C3_true, [0, 1], axis=1)
    C3_predicted = np.delete(C3_predicted, [0,1], axis=1)

    C1_recall = get_recall(C1_true, C1_predicted)
    C2_recall = get_recall(C2_true, C2_predicted)
    C3_recall = get_recall(C3_true, C3_predicted)

    C1_specificity = get_specificity(C1_true, C1_predicted)
    C2_specificity = get_specificity(C2_true, C2_predicted)
    C3_specificity = get_specificity(C3_true, C3_predicted)

    C1_balanced_accuracy = (C1_recall+C1_specificity)/2
    C2_balanced_accuracy = (C2_recall+C2_specificity)/2
    C3_balanced_accuracy = (C3_recall+C3_specificity)/2
    total_balanced_accuracy = (C1_recall+C2_recall+C3_recall)/3

    return C1_balanced_accuracy, C2_balanced_accuracy, C3_balanced_accuracy, total_balanced_accuracy

def get_recall(true, predicted):
    TP = np.sum((true == 1) & (predicted == 1))
    FN = np.sum((true == 1) & (predicted == 0))
    return TP / (TP + FN)


def get_specificity(true, predicted):
    TN = np.sum((true == 0) & (predicted == 0))
    FP = np.sum((true == 0) & (predicted == 1))
    return TN / (TN + FP)


def get_map(y_true, y_pred_probs_list):
    average_precisions = []

    true_labels = np.concatenate([np.array(x) for x in y_true])
    predicted_probabilities = np.concatenate([np.array(x) for x in y_pred_probs_list])
    for class_idx in range(true_labels.shape[1]):
        class_true = true_labels[:, class_idx]
        class_scores = predicted_probabilities[:, class_idx]
        average_precision = average_precision_score(class_true, class_scores)
        average_precisions.append(average_precision)

    # Calculate the mean of the average precisions across all classes to obtain mAP
    mAP = np.mean(average_precisions)
    C1_ap = average_precisions[0]
    C2_ap = average_precisions[1]
    C3_ap = average_precisions[2]
    
    return C1_ap, C2_ap, C3_ap, mAP

def find_best_epoch(results_dict):
    best_result = 0
    for key, val in results_dict.items():
        map_score = results_dict[key]['avg_map']
        if map_score >= best_result:
            best_epoch = key
            best_result = map_score
    return best_epoch

def check_memory():
    # Get the current GPU device
    device = torch.cuda.current_device()
    
    # Get the total and allocated memory
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    reserved_memory = torch.cuda.memory_reserved(device)

    # Print the memory information
    print(f"Total Memory: {total_memory / (1024**2):.2f} MB")
    print(f"Allocated Memory: {allocated_memory / (1024**2):.2f} MB")
    print(f"Reserved Memory: {reserved_memory / (1024**2):.2f} MB\n")