#This file creates the neccesary json files (following endoscapes format) using challenge data

import os
import json
import pandas as pd


def save_json_file(path: str, data: dict):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_json_file(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def join_rater_annots(csv_path: str, fold:int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function loads the csv, join all the annotations in just 3 columns with the mean value for each criteria
    """
    df = pd.read_csv(csv_path)
    for c in ['c1', 'c2', 'c3']:
        cols = [col for col in df.columns if col.startswith(c)]
        df[f'mean_{c}'] = df[cols].mean(axis=1)

    df['ds'] = df[['mean_c1', 'mean_c2', 'mean_c3']].values.tolist()
    
    df['video_id'] = df['Video_name'].str.extract(r'video_(\d+)')
    df['file_name'] = df['video_id'].astype(str) + '_' + df['frame_id'].astype(str) + '.jpg'

    #Remove unnecesary columns
    # Construir lista de columnas a eliminar
    cols_to_drop = ['video_id', 'frame_id', 'Video_name']

    # Agregar rater columns y mean columns
    for i in range(1, 4):
        cols_to_drop += [f'c{i}_rater{j}' for j in range(1, 4)]
        cols_to_drop.append(f'mean_c{i}')

    # Hacer dos dataframes dependiendo del split al que pertenecen
    splits_data = load_json_file(f'../Sages/Splits_partition/fold{fold}_video_splits.json')
    train_videos, test_videos = splits_data['train'], splits_data['test']

    df_train = df[df['Video_name'].isin(train_videos)]
    df_test = df[df['Video_name'].isin(test_videos)]
    # Eliminar columnas (sin error si alguna no existe)
    df_train.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    df_test.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return df_train, df_test



if __name__ == '__main__':
    fold = 1
    original_sages_data_path = 'Sages'
    sages_annotations_file = f'../{original_sages_data_path}/labels/unified_frame_labels.csv'
    df_train, df_test = join_rater_annots(sages_annotations_file, fold=fold)

    for split, df in zip(['train', 'test'],[df_train, df_test]):
        list_of_dicts = df[['file_name', 'ds']].to_dict(orient='records')

        data_dict = {'images': list_of_dicts}
        save_json_file(f'Sages_fold{fold}_{split}_data.json' ,
                    data_dict)

