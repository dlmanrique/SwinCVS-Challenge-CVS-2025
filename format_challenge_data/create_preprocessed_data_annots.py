# This file verfies that the keyframes in the original Sages dataset exists as preprocessed frames, if not just delete it from the keyframes files

import os
import json
import glob


def save_json_file(path: str, data: dict):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_json_file(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def key_frames_verifier(pth: str):
    """
    This function expects the path to one json file, then it iterate over all the keyframes and verify that they exists in frames_cuitmargins.
    Then creates a equivalent json only with the frames that exists
    """

    data = load_json_file(pth)
    frame_cut_margin_folder_path = '../Sages/frames_cutmargin'
    new_info = []
    for img_info in data['images']:

        video_id, frame_id = img_info['file_name'].split('_')[0], img_info['file_name'].split('_')[1][:-4].zfill(5)
        frame_path = os.path.join(frame_cut_margin_folder_path, f'video_{video_id}', f'{frame_id}.jpg')

        if os.path.exists(frame_path):
            new_info.append(img_info)
        else: 
            print(f'Non existent file: {frame_path}')

    fold = pth.split('fold')[-1][0]
    split = pth.split('_')[2]
    save_path = f'preprocessed_data/Fold{fold}'
    os.makedirs(save_path, exist_ok=True)

    save_json_file(os.path.join(save_path, f'{split}.json'), {'images': new_info})

    return len(new_info)
    

if __name__ == '__main__':
    original_files_paths = glob.glob('*.json')

    for pth in original_files_paths:
        print(f'Verifing file : {pth}')
        number_frames = key_frames_verifier(pth)
        print(f'Number of frames resulting on the new file: {number_frames}')
        
