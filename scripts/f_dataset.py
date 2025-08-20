import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import json
import pandas as pd
import shutil
import wandb

from torchvision import transforms
from pathlib import Path
from scripts.f_environment import download_extract_zip

def get_datasets(config, args):
    """
    Check dataset exists (download if not). Create dataset instances and apply transformations specified in config
    """

    if config.DATASET == 'Endoscapes':
        dataset_dir = check_dataset(config)
    elif config.DATASET == 'Sages':
        dataset_dir = config.DATASET_DIR
    
    print(f"\nDataset loaded from: {dataset_dir}")

    train_dataframe, val_dataframe, test_dataframe = get_three_dataframes(dataset_dir, config, args, lstm=config.MODEL.LSTM)


    print(f'Number of keyframes on train split: {len(train_dataframe)}')
    print(f'Number of keyframes on valid split: {len(val_dataframe)}')
    print(f'Number of keyframes on test split: {len(test_dataframe)}')

    if args.direction != 'None':
        # Sanity check info
        train_file = f'format_challenge_data/extended_annots/{args.extend_method}/Fold{config.FOLD}/{args.direction}/{args.fps}/train_data.json'
        val_file = f'format_challenge_data/extended_annots/{args.extend_method}/Fold{config.FOLD}/{args.direction}/{args.fps}/test_data.json'
        test_file = f'format_challenge_data/extended_annots/{args.extend_method}/Fold{config.FOLD}/{args.direction}/{args.fps}/test_data.json'

        wandb.log({'Train data': train_file})
        wandb.log({'Val data': val_file})
        wandb.log({'Test data': test_file})


    transform_sequence = get_transform_sequence(config)
    
    # If SwinCVS model
    if config.MODEL.LSTM:
        training_dataset = EndoscapesSwinCVS_Dataset(train_dataframe[::config.TRAIN.LIMIT_DATA_FRACTION], transform_sequence)
        val_dataset = EndoscapesSwinCVS_Dataset(val_dataframe[::config.TRAIN.LIMIT_DATA_FRACTION], transform_sequence)
        test_dataset = EndoscapesSwinCVS_Dataset(test_dataframe[::config.TRAIN.LIMIT_DATA_FRACTION], transform_sequence)

    # If just SwinV2 backbone
    else:
        training_dataset = Endoscapes_Dataset(train_dataframe[::config.TRAIN.LIMIT_DATA_FRACTION], transform_sequence)
        val_dataset = Endoscapes_Dataset(val_dataframe[::config.TRAIN.LIMIT_DATA_FRACTION], transform_sequence)
        test_dataset = Endoscapes_Dataset(test_dataframe[::config.TRAIN.LIMIT_DATA_FRACTION], transform_sequence)

    return training_dataset, val_dataset, test_dataset

def check_dataset(config):
    """
    Checks whether specified folder contains valid endoscapes dataset. Redownloads if checksum failed, or folder missing.
    Requires config.DATASET_DIR to lead to the folder containing 'endoscapes' or null - will download to repo dir. 
    """

    dataset_path = config.DATASET_DIR

    # If dataset is meant to be downloaded into cwd
    if dataset_path == None:
        dataset_path = Path.cwd()

    # Add 'endoscapes' subfolder
    dataset_dir = Path(dataset_path) / 'endoscapes'

    # Checksum of the number of expected files
    all_imgs_dir = dataset_dir / 'all'
    if all_imgs_dir.exists() and all_imgs_dir.is_dir():
        file_count = sum(1 for f in all_imgs_dir.iterdir() if f.is_file())
        if file_count != 58586:
            response = input(f"Dataset checksum failed. Attempting to remove the '{dataset_dir}' and redownload Endoscapes dataset. Proceed? (Y/N): ").strip().upper()
            while response not in ['Y', 'N']:
                input(f"Please answer with Y/N only")
            if response == 'Y':
                print('Removing pre-existing dataset...')
                shutil.rmtree(dataset_dir)
                print('Re-downloading dataset')
                download_extract_zip(dataset_dir.parent, 'https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip')
            if response == 'N':
                print("Continuing with the originally specified dataset...")
    else:
        print('Dataset folder not found. Downloading dataset...')
        if dataset_dir.exists() and dataset_dir.is_dir():
            shutil.rmtree(dataset_dir)
        print('Dataset downloaded. Unpacking...')
        download_extract_zip(dataset_dir.parent, 'https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip')
    return dataset_dir

def get_dataloaders(config, training_dataset, val_dataset, test_dataset):
    """
    Create dataloaders from a given training datasets
    """
    print(f"Batch size: {config.TRAIN.BATCH_SIZE}")
    train_dataloader = DataLoader(  training_dataset,
                                    batch_size = config.TRAIN.BATCH_SIZE,
                                    pin_memory = True,
                                    shuffle = True)

    val_dataloader = DataLoader(    val_dataset,
                                    batch_size = 1,
                                    shuffle = False,
                                    pin_memory = True)

    test_dataloader = DataLoader(   test_dataset,
                                    batch_size = 1,
                                    shuffle = False,
                                    pin_memory = True)
    return train_dataloader, val_dataloader, test_dataloader

def get_three_dataframes(image_folder, config, args, lstm = False):
    """
    Get images from the dataset directory, create pandas dataframes of image filepaths and ground truths. 
    """

    if config.DATASET == 'Endoscapes':
        # Specify directories for the splits
        train_dir = image_folder / 'train'
        val_dir  = image_folder / 'val'
        test_dir = image_folder / 'test'

        # Get filepaths for individual images
        train_file = [x for x in os.listdir(train_dir) if 'json' and 'ds_coco' in x][0]
        val_file = [x for x in os.listdir(val_dir) if 'json' and 'ds_coco' in x][0]
        test_file = [x for x in os.listdir(test_dir) if 'json' and 'ds_coco' in x][0]

        # Create dataframe with filepaths for individual images along with ground truth labels
        train_dataframe = get_dataframe(train_dir / train_file)
        val_dataframe = get_dataframe(val_dir / val_file)
        test_dataframe = get_dataframe(test_dir / test_file)

    elif config.DATASET == 'Sages':

        if args.extend_method != 'None':
            # If exists, then the experiment with extending annots
            train_file = f'format_challenge_data/extended_annots/{args.extend_method}/Fold{config.FOLD}/{args.direction}/{args.fps}/train_data.json'
            val_file = f'format_challenge_data/extended_annots/{args.extend_method}/Fold{config.FOLD}/{args.direction}/{args.fps}/test_data.json'
            test_file = f'format_challenge_data/extended_annots/{args.extend_method}/Fold{config.FOLD}/{args.direction}/{args.fps}/test_data.json'
        
        else:

            # Mapear rutas según tipo de frame
            train_paths = {
                "Original": f"format_challenge_data/Sages_fold{config.FOLD}_train_data.json",
                "Preprocessed": f"format_challenge_data/preprocessed_data/Fold{config.FOLD}/train.json"
            }

            test_paths = {
                "Original": f"format_challenge_data/Sages_fold{config.FOLD}_test_data.json",
                "Preprocessed": f"format_challenge_data/preprocessed_data/Fold{config.FOLD}/test.json"
            }

            # Seleccionar según args
            train_file = train_paths.get(args.frame_type_train)
            val_file = test_paths.get(args.frame_type_test)
            test_file = test_paths.get(args.frame_type_test)


        # Create dataframe with filepaths for individual images along with ground truth labels
        train_dataframe = get_dataframe(train_file)
        val_dataframe = get_dataframe(val_file)
        test_dataframe = get_dataframe(test_file)


    if lstm:
        # Add unlabelled images to the dataframe
        with open(image_folder / 'all' / 'annotation_coco.json', 'r') as file:
            all_images = json.load(file)  # Load JSON data
        all_image_names = [x['file_name'] for x in all_images['images']]

        # Add images to correct file lists according from which video they are
        train_images = [img for img in all_image_names if 1 <= int(img.split('_')[0]) <= 120]
        val_images = [img for img in all_image_names if 121 <= int(img.split('_')[0]) <= 161]
        test_images = [img for img in all_image_names if 162 <= int(img.split('_')[0]) <= 201]

        # Adding unlabelled images
        train_dataframe = add_unlabelled_imgs(train_images, train_dataframe)
        val_dataframe = add_unlabelled_imgs(val_images, val_dataframe)
        test_dataframe = add_unlabelled_imgs(test_images, test_dataframe)
        
        # Generate 5 frame sequences and update format to include paths to images
        train_dataframe = get_frame_sequence_dataframe(train_dataframe, train_dir)
        val_dataframe = get_frame_sequence_dataframe(val_dataframe, val_dir)
        test_dataframe = get_frame_sequence_dataframe(test_dataframe, test_dir)
        return train_dataframe, val_dataframe, test_dataframe

    updated_train_dataframe = update_dataframe(train_dataframe, config.DATASET_DIR, config, args, 'train')
    updated_val_dataframe = update_dataframe(val_dataframe, config.DATASET_DIR, config, args, 'valid')
    updated_test_dataframe = update_dataframe(test_dataframe, config.DATASET_DIR, config, args, 'test')
    

    return updated_train_dataframe, updated_val_dataframe, updated_test_dataframe

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
      
        return image, label, image_path
    
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

def update_dataframe(dataframe, image_folder, config, args, split):
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

        if args.frame_type_train == 'Original':
            image_folder = os.path.join(image_folder, 'frames')

        elif args.frame_type_train == 'Preprocessed' and split == 'train':
            # Establezco que el train siempre sea preprocesado, lo que varia es el test
            image_folder = os.path.join(image_folder, 'frames_cutmargin')
        
        elif args.frame_type_test == 'Preprocessed' and (split == 'valid' or split == 'test'):
            image_folder = os.path.join(image_folder, 'frames_cutmargin')

        elif args.frame_type_test == 'Original' and (split == 'valid' or split == 'test'):
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