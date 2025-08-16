# This file extends the CVS annotations in two ossible ways: all annots or only positive annotations

import os
import json
import glob
import pandas as pd
from tqdm import tqdm


def save_json_file(path: str, data: dict):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_json_file(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extend_annots(dict_annots: dict, fold: str, split: str, fps: int, balanced: bool, times: int, direction: str):
    """"
    En este caso dejamos que nos llega el json normal, la cantidad de fps que queremos prolongar esa anotacion y si queremos que solo sea con
    las positivas o con todas (balanced=True -> lo hago con todas). La variable de times me dice cuantas veces voy a prolongar una annot, esto es que
    si tengo 1 es porque tengo el doble de datos siempre que tenga balanced True. El parametro de direction me dice hacia donde hago la extension
    de la annot, este tiene 3 opciones: 'future', 'past' and 'both'
    """

    new_info = []
    max_frame_id = 2700  # límite superior del frame_id
    min_frame_id = 1

    for idx in tqdm(range(len(dict_annots['images']))):
        original_frame_info = dict_annots['images'][idx]
        frame_name, annot = original_frame_info['file_name'], original_frame_info['ds']
        video_id, frame_id = int(frame_name.split('_')[0]), int(frame_name.split('_')[1][:-4])

        # Add original info
        new_info.append({'file_name': f'{str(video_id).zfill(3)}_{frame_id}.jpg',
                         'ds': annot})

  
        # Extend annotations depending on function parameters
        if balanced:
            for t in range(1, times + 1):
                offset = fps * t

                if direction in ('future', 'both'):
                    future_id = frame_id + offset
                    if future_id <= max_frame_id:
                        new_info.append({'file_name': f'{str(video_id).zfill(3)}_{future_id}.jpg',
                                        'ds': annot})

                if direction in ('past', 'both'):
                    past_id = frame_id - offset
                    if past_id >= min_frame_id:
                        new_info.append({'file_name': f'{str(video_id).zfill(3)}_{past_id}.jpg',
                                        'ds': annot})


        else:
            if sum(annot) > 0:
                for t in range(1, times + 1):
                    offset = fps * t

                    if direction in ('future', 'both'):
                        future_id = frame_id + offset
                        if future_id <= max_frame_id:
                            new_info.append({'file_name': f'{str(video_id).zfill(3)}_{future_id}.jpg',
                                            'ds': annot})

                    if direction in ('past', 'both'):
                        past_id = frame_id - offset
                        if past_id >= min_frame_id:
                            new_info.append({'file_name': f'{str(video_id).zfill(3)}_{past_id}.jpg',
                                            'ds': annot})
                

    balanced_str = 'balanced' if balanced else 'unbalanced'
    save_dir = f'extended_annots/{balanced_str}/Fold{fold}/{direction}/{fps}'
    os.makedirs(save_dir, exist_ok=True)
    save_json_file(os.path.join(save_dir, f'{split}_data.json'), {'images': new_info})



if __name__ == '__main__':
    
    json_files = glob.glob(os.path.join('*.json'))


    for json_name in json_files:
        data = load_json_file(json_name)
        fold = json_name.split('fold')[-1][0]
        split = json_name.split('_')[2]

        extend_annots(data,
                      fold,
                      split,
                      fps=30, 
                      balanced=False, 
                      times=1,
                      direction='future')