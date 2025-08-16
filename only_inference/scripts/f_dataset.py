import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import json
import pandas as pd
from torchvision import transforms



def get_datasets(config):
    """
    Check dataset exists (download if not). Create dataset instances and apply transformations specified in config
    """
    breakpoint()
    dataset_dir = config.DATASET_DIR
    
    print(f"\nDataset loaded from: {dataset_dir}")

    test_dataframe = get_dataframe_test(config)
    transform_sequence = get_transform_sequence(config)
    
    # If just SwinV2 backbone
    test_dataset = Endoscapes_Dataset(test_dataframe, transform_sequence)

    return test_dataset


def get_dataloaders(config, test_dataset):
    """
    Create dataloaders from a given training datasets
    """
    print(f"Batch size: {config.TRAIN.BATCH_SIZE}")

    test_dataloader = DataLoader(   test_dataset,
                                    batch_size = 1,
                                    shuffle = False,
                                    pin_memory = True)
    return test_dataloader


def get_dataframe_test(config):
    """
    Get images from the dataset directory, create pandas dataframes of image filepaths and ground truths. 
    """
    test_file = config.TEST_FILE
    test_dataframe = get_dataframe(test_file)
    updated_test_dataframe = update_dataframe(test_dataframe, config.DATASET_DIR, config)
    
    return updated_test_dataframe


class Endoscapes_Dataset(Dataset):
    """
    Dataset creator only for backbone - SwinV2 training.
    """
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
            image = (image-torch.min(image)) / (-torch.min(image)+torch.max(image)) #Normalize the image in the interval (0,1)
      
        return image, label



class EndoscapesSwinCVS_Dataset(Dataset):
    """
    Dataset creator for SwinCVS - includes 5 frame sequences.
    """
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
            seed = random.randint(0, 2**32)
            for path in paths:
                image = Image.open(path)
                torch.manual_seed(seed)
                random.seed(seed)
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


def get_dataframe(json_path):
    """
    Get dataframes of the dataset splits in columns:
    idx | vid | frame | C1 | C2 | C3
    """

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
        C1_i = round(i['ds'][0]) #Con esto tienen en cuenta el tema de 3 anotadores
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


def get_frame_sequence_dataframe(dataframe, image_folder):
    """
    For LSTM dataframe creator. Using dataframes updated with unlabelled images, get five frame sequences. The returned dataframe has columns:
    idx | f0 | f1 | f2 | f3 | f4 | classification
    idx - index of the sequence
    f0-4 - path to each image in the sequence
    classification - list of ground truth values for C1-3 as, [C1, C2, C3] e.g. [0.0, 0.0, 1.0] 
    """

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


def update_dataframe(dataframe, image_folder, config):
    """
    Function only for creation of dataframes when training backbone - SwinV2. It changes the structure of the dataframe from:
    idx | vid | frame | C1 | C2 | C3
    to:
    idx | path | classification
    where path is a path to a given image and classification is a list of ground truth values for C1-3 as, [C1, C2, C3] e.g. [0.0, 0.0, 1.0] 
    """
    
    
    if config.DATASET == 'Endoscapes':
        dataframe['path'] = dataframe.apply(lambda row: generate_path(row, image_folder), axis=1)
    elif config.DATASET == 'Sages':
        image_folder = os.path.join(image_folder, 'frames')
        dataframe['path'] = dataframe.apply(lambda row: generate_path_sages(row, image_folder), axis=1)

    dataframe['classification'] = dataframe.apply(lambda row: get_class(row), axis=1)
    dataframe.drop(columns=['vid', 'frame', 'C1', 'C2', 'C3'], inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe

def add_unlabelled_imgs(list_of_selected_images, selected_dataframe):
    """
    Given existing splits dataframes, in correct otder, append images that are unlabelled. Output dataframe has columns:
    idx | vid | frame | C1 | C2 | C3
    where C1-3 is unlabelled it has a value '-1.0'
    """
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

def generate_path(row, image_folder):
    vid = row['vid']
    frame = row['frame']
    filename = str(vid) + '_' + str(frame) + '.jpg'
    path = os.path.join(image_folder, filename)
    return str(path)

def generate_path_sages(row, image_folder):

    vid = row['vid']
    frame = row['frame']
    filename = 'video_' + str(vid).zfill(3) + '/' + str(frame).zfill(5) + '.jpg'
    path = os.path.join(image_folder, filename)
    return str(path)

def get_class(row):
    classification = [float(row['C1']), float(row['C2']), float(row['C3'])]
    return classification

def get_endoscapes_mean_std(config):
    mean = config.TRAIN.TRANSFORMS.ENDOSCAPES_MEAN
    std = config.TRAIN.TRANSFORMS.ENDOSCAPES_STD

    # Change BGR to RGB
    mean = mean[::-1]
    std = std[::-1]

    return mean, std


def get_transform_sequence(config):
    mean, std = get_endoscapes_mean_std(config)
    transform_sequence = transforms.Compose([   transforms.CenterCrop(config.TRAIN.TRANSFORMS.CENTER_CROP),
                                                transforms.Resize((384, 384)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(
                                                    mean=torch.tensor(mean),
                                                    std=torch.tensor(std))])
    return transform_sequence