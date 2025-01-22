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
from PIL import Image, ImageOps
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

# FUNCTIONS

def read_config(config_file):
    """
    Read Yaml file into dict
    """
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def config_to_yacs(dict_config):
    """
    Convert the dict to a yacs CfgNode.
    """
    if not isinstance(dict_config, dict):
        return dict_config  # Return non-dict values as is
    cfg = CN()
    for key, value in dict_config.items():
        cfg[key] = config_to_yacs(value)  # Recursively convert nested dictionaries
    return cfg

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


def build_optimizer(config, model, multiclass = False, **kwargs):
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    parameters, no_decay_names = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None

    if multiclass:
        parameters2= [ {'params': model.swinv2_model.parameters(), 'lr': config.TRAIN.OPTIMIZER.ENCODER_LR},
                    {'params': model.fc_swin.parameters(), 'lr': config.TRAIN.OPTIMIZER.ENCODER_LR},
                    {'params': model.lstm.parameters(), 'lr': config.TRAIN.OPTIMIZER.CLASSIFIER_LR},
                    {'params': model.fc_lstm.parameters(), 'lr': config.TRAIN.OPTIMIZER.CLASSIFIER_LR}]
    else:
        parameters2= [ {'params': model.swinv2_model.parameters(), 'lr': config.TRAIN.OPTIMIZER.ENCODER_LR},
                        {'params': model.lstm.parameters(), 'lr': config.TRAIN.OPTIMIZER.CLASSIFIER_LR },
                        {'params': model.fc.parameters(), 'lr': config.TRAIN.OPTIMIZER.CLASSIFIER_LR}]



    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
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

def get_three_dataframes(image_folder, lstm = False):
    train_dir = image_folder / 'train'
    val_dir  = image_folder / 'val'
    test_dir = image_folder / 'test'
    train_file = [x for x in os.listdir(train_dir) if 'json' and 'ds_coco' in x][0]
    val_file = [x for x in os.listdir(val_dir) if 'json' and 'ds_coco' in x][0]
    test_file = [x for x in os.listdir(test_dir) if 'json' and 'ds_coco' in x][0]

    train_dataframe = get_dataframe(train_dir / train_file)
    val_dataframe = get_dataframe(val_dir / val_file)
    test_dataframe = get_dataframe(test_dir / test_file)

    if lstm:
        # Add unlabelled images to the dataframe
        with open(image_folder / 'all' / 'annotation_coco.json', 'r') as file:
            all_images = json.load(file)  # Load JSON data
        all_image_names = [x['file_name'] for x in all_images['images']]

        train_images = [img for img in all_image_names if 1 <= int(img.split('_')[0]) <= 120]
        val_images = [img for img in all_image_names if 121 <= int(img.split('_')[0]) <= 161]
        test_images = [img for img in all_image_names if 162 <= int(img.split('_')[0]) <= 201]
        print('$Dataframes$: Adding unlabelled images...')
        train_dataframe = add_unlabelled_imgs(train_images, train_dataframe)
        val_dataframe = add_unlabelled_imgs(val_images, val_dataframe)
        test_dataframe = add_unlabelled_imgs(test_images, test_dataframe)

        # Generate 5 frame sequences and update format to include paths to images
        print('$Dataframes$: Updating dataframes...')
        train_dataframe = get_frame_sequence_dataframe(train_dataframe, train_dir)
        val_dataframe = get_frame_sequence_dataframe(val_dataframe, val_dir)
        test_dataframe = get_frame_sequence_dataframe(test_dataframe, test_dir)
        return train_dataframe, val_dataframe, test_dataframe

    updated_train_dataframe = update_dataframe(train_dataframe, train_dir)
    updated_val_dataframe = update_dataframe(val_dataframe, val_dir)
    updated_test_dataframe = update_dataframe(test_dataframe, test_dir)

    return updated_train_dataframe, updated_val_dataframe, updated_test_dataframe

def get_dataframe(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    vid = []
    frame = []
    C1 = []
    C2 = []
    C3 = []

    for i in data['images']:
        # Extract data
        file_name = i['file_name']
        file_name = file_name.split('.')[0]
        file_name = file_name.split('_')
        vid_i = file_name[0]
        frame_i = file_name[1]
        C1_i = round(i['ds'][0])
        C2_i = round(i['ds'][1])
        C3_i = round(i['ds'][2])

        # Put in list
        vid.append(vid_i)
        frame.append(frame_i)
        C1.append(C1_i)
        C2.append(C2_i)
        C3.append(C3_i)

    data_dict = {'vid': vid,
                'frame': frame,
                'C1': C1,
                'C2': C2,
                'C3': C3}
    data_dataframe = pd.DataFrame(data_dict)
    return data_dataframe

def update_dataframe(dataframe, image_folder):
    dataframe['path'] = dataframe.apply(lambda row: generate_path(row, image_folder), axis=1)
    dataframe['classification'] = dataframe.apply(lambda row: get_class(row), axis=1)
    dataframe.drop(columns=['vid', 'frame', 'C1', 'C2', 'C3'], inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe

def add_unlabelled_imgs(list_of_selected_images, selected_dataframe):
    rows = []
    for image in list_of_selected_images:
        contents = image.split('.')[0].split('_')
        frame_info = (contents[0], contents[1])
        rows.append({'vid': frame_info[0], 'frame': frame_info[1], 'C1': -1, 'C2': -1, 'C3': -1})

    df = pd.DataFrame(rows)

    combined_df = pd.merge(df, selected_dataframe, on=['vid', 'frame'], how='left', suffixes=('_new', '_lbld'))
    combined_df['C1'] = combined_df['C1_lbld'].combine_first(combined_df['C1_new'])
    combined_df['C2'] = combined_df['C2_lbld'].combine_first(combined_df['C2_new'])
    combined_df['C3'] = combined_df['C3_lbld'].combine_first(combined_df['C3_new'])

    # Drop the redundant columns from df1
    final_df = combined_df[['vid', 'frame', 'C1', 'C2', 'C3']]


    final_df = final_df.sort_values(by=['vid', 'frame'])
    final_df = final_df.reset_index(drop=True)

    return final_df

def get_frame_sequence_dataframe(dataframe, image_folder):
    new_dataframe_rows = []
    # Iterate over each video so as not to create intravid sequences
    for video in dataframe['vid'].unique():
        temp_vid_dataframe = dataframe.loc[dataframe['vid'] == video]
        # Iterate over each datapoint in the dataframe
        for idx in range(len(temp_vid_dataframe)-5):
            # Extract 5 frame sequences
            five_seq_dataframe = temp_vid_dataframe.iloc[idx:idx+5]

            # Check if the last frame in the sequence is labelled
            if five_seq_dataframe.iloc[4]['C1'] != -1:
                # Update paths to images for all five frames
                paths = []
                for datapoint in five_seq_dataframe.iterrows():
                    paths.append(generate_path(datapoint[1], image_folder)) # CHANGE VAL_DIR!!!!
                # Get class of the last frame
                classification = get_class(five_seq_dataframe.iloc[4])
                # Put it in a new row of the dataframe
                new_row = { 'f0': paths[0], 'f1':  paths[1], 'f2': paths[2], 'f3': paths[3], 'f4': paths[4],
                            'classification': classification}
                new_dataframe_rows.append(new_row)

    updated_dataframe = pd.DataFrame(new_dataframe_rows)
    
    return updated_dataframe

class EndoscapesSwinLSTM_Dataset3imgs(Dataset):
    def __init__(self, image_dataframe, transform_sequence):
        self.image_dataframe = image_dataframe
        self.transforms = transform_sequence
        
    def __len__(self):
        return len(self.image_dataframe)
    
    def __getitem__(self, idx):
        sequence_info = self.image_dataframe.iloc[idx]
        image_f2_path = sequence_info['f2']
        image_f3_path = sequence_info['f3']
        image_f4_path = sequence_info['f4']
        paths = [image_f2_path, image_f3_path, image_f4_path]
        
        image_list = []
        if self.transforms:
            seed = random.randint(0, 2**32) # Added
            for path in paths:
                image = Image.open(path)
                torch.manual_seed(seed) # Added
                random.seed(seed) # Added
                image = self.transforms(image)
                image = (image-torch.min(image)) / (-torch.min(image)+torch.max(image))
                image_list.append(image)
        else:
            for path in paths:
                image = Image.open(path)
                image_list.append(image)
        
        images = torch.stack(image_list)
        label = torch.tensor(sequence_info['classification'])

        return images, label
    
def get_frame_sequence_dataframe3imgs(dataframe, image_folder):
    new_dataframe_rows = []
    # Iterate over each video so as not to create intravid sequences
    for video in dataframe['vid'].unique():
        temp_vid_dataframe = dataframe.loc[dataframe['vid'] == video]
        # Iterate over each datapoint in the dataframe
        for idx in range(len(temp_vid_dataframe)-5):
            # Extract 5 frame sequences
            five_seq_dataframe = temp_vid_dataframe.iloc[idx:idx+5]

            # Check if the last frame in the sequence is labelled
            if five_seq_dataframe.iloc[4]['C1'] != -1:
                # Update paths to images for all five frames
                paths = []
                for datapoint in five_seq_dataframe.iterrows():
                    paths.append(generate_path(datapoint[1], image_folder))
                # Get class of the last frame
                classification = get_class(five_seq_dataframe.iloc[4])
                # Put it in a new row of the dataframe
                new_row = { 'f2': paths[2], 'f3': paths[3], 'f4': paths[4], 'classification': classification}
                new_dataframe_rows.append(new_row)

    updated_dataframe = pd.DataFrame(new_dataframe_rows)
    
    return updated_dataframe

def get_three_dataframes3imgs(image_folder, lstm = False):
    train_dir = image_folder / 'train'
    val_dir  = image_folder / 'val'
    test_dir = image_folder / 'test'
    train_file = [x for x in os.listdir(train_dir) if 'json' and 'ds_coco' in x][0]
    val_file = [x for x in os.listdir(val_dir) if 'json' and 'ds_coco' in x][0]
    test_file = [x for x in os.listdir(test_dir) if 'json' and 'ds_coco' in x][0]

    train_dataframe = get_dataframe(train_dir / train_file)
    val_dataframe = get_dataframe(val_dir / val_file)
    test_dataframe = get_dataframe(test_dir / test_file)

    if lstm:
        # Add unlabelled images to the dataframe
        with open(image_folder / 'all' / 'annotation_coco.json', 'r') as file:
            all_images = json.load(file)  # Load JSON data
        all_image_names = [x['file_name'] for x in all_images['images']]

        train_images = [img for img in all_image_names if 1 <= int(img.split('_')[0]) <= 120]
        val_images = [img for img in all_image_names if 121 <= int(img.split('_')[0]) <= 161]
        test_images = [img for img in all_image_names if 162 <= int(img.split('_')[0]) <= 201]
        print('$Dataframes$: Adding unlabelled images...')
        train_dataframe = add_unlabelled_imgs(train_images, train_dataframe)
        val_dataframe = add_unlabelled_imgs(val_images, val_dataframe)
        test_dataframe = add_unlabelled_imgs(test_images, test_dataframe)

        # Generate 5 frame sequences and update format to include paths to images
        print('$Dataframes$: Updating dataframes...')
        train_dataframe = get_frame_sequence_dataframe3imgs(train_dataframe, train_dir)
        val_dataframe = get_frame_sequence_dataframe3imgs(val_dataframe, val_dir)
        test_dataframe = get_frame_sequence_dataframe3imgs(test_dataframe, test_dir)
        return train_dataframe, val_dataframe, test_dataframe

    updated_train_dataframe = update_dataframe(train_dataframe, train_dir)
    updated_val_dataframe = update_dataframe(val_dataframe, val_dir)
    updated_test_dataframe = update_dataframe(test_dataframe, test_dir)

    return updated_train_dataframe, updated_val_dataframe, updated_test_dataframe


def generate_path(row, image_folder):
    vid = row['vid']
    frame = row['frame']
    filename = str(vid) + '_' + str(frame) + '.jpg'
    path = os.path.join(image_folder, filename)
    return str(path)

def get_class(row):
    classification = [float(row['C1']), float(row['C2']), float(row['C3'])]
    return classification
    
class Endoscapes_Dataset(Dataset):
    def __init__(self, image_dataframe, transform_sequence):
        self.image_dataframe = image_dataframe
        self.transforms = transform_sequence
        
    def __len__(self):
        return len(self.image_dataframe)
    
    def __getitem__(self, idx):
        image_info = self.image_dataframe.iloc[idx]
        image_path = image_info['path']
        label = torch.tensor(image_info['classification'])

        image = Image.open(image_path)
        
        if self.transforms:
            image = self.transforms(image)
            image = (image-torch.min(image)) / (-torch.min(image)+torch.max(image))
      
        return image, label
    
class EndoscapesSwinLSTM_Dataset(Dataset):
    def __init__(self, image_dataframe, transform_sequence):
        self.image_dataframe = image_dataframe
        self.transforms = transform_sequence
        
    def __len__(self):
        return len(self.image_dataframe)
    
    def __getitem__(self, idx):
        sequence_info = self.image_dataframe.iloc[idx]
        image_f0_path = sequence_info['f0']
        image_f1_path = sequence_info['f1']
        image_f2_path = sequence_info['f2']
        image_f3_path = sequence_info['f3']
        image_f4_path = sequence_info['f4']
        paths = [image_f0_path, image_f1_path, image_f2_path, image_f3_path, image_f4_path]
        
        image_list = []
        if self.transforms:
            seed = random.randint(0, 2**32) # Added
            for path in paths:
                image = Image.open(path)
                torch.manual_seed(seed) # Added
                random.seed(seed) # Added
                image = self.transforms(image)
                image = (image-torch.min(image)) / (-torch.min(image)+torch.max(image))
                image_list.append(image)
        else:
            for path in paths:
                image = Image.open(path)
                image_list.append(image)
        
        images = torch.stack(image_list)
        label = torch.tensor(sequence_info['classification'])

        return images, label
    

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