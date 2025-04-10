import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Manager
from utils.preprocessing.visual_data_preprocessing import get_video_creation_time
from utils.preprocessing.physiological_signal_preprocessing import get_ECGnPPG_creation_time
import re


def preprocess(Config):
    
    """
        Data folder structure should be:
        ****** All folder names must be in lower case and singular notation ******
            root
                video (or image)
                
                    160_1.mp4
                    160_2.mp4
                    ...
                    16n_n.mp4
                    
                image (The images should simply be extracted frames from the video)
                    160_1
                        000.png
                        001.png
                        ...
                        nnn.png
                    160_2
                        000.png
                        001.png
                        ...
                        nnn.png
                    
                    ...
                    
                    16n_n
                        000.png
                        001.png
                        ...
                        nnn.png
                    
                eeg
                    160_1.mat
                    160_2.mat
                    ...
                    16n_n.mat
                    
                polar (ecg)
                    160_1.csv
                    160_2.csv
                    ...
                    16n_n.csv
                    
                rppg (if exist)
                    160_1.npy
                    160_2.npy
                    ...
                    16n_n.npy
    """
    
    if not isinstance(Config.data_path, str):
        raise TypeError(f"data path: {Config.data_path} is not a string")
    
    if not isinstance(Config.cache_path, str):
        raise TypeError(f"save path: {Config.save_path} is not a string")
    
    match_files(Config)
    
    physiological_time_line = get_ECGnPPG_creation_time(Config)
    visual_time_line = get_video_creation_time(Config)
    
    
    
    if 'video' in Config.input_features and Config.multi_process == 1:
        print("\nStarting video preprocessing")
        video_preprocess(Config)
    elif 'video' in Config.input_features and Config.multi_process >= 2:
        print("\nStarting video preprocessing with multiprocess")
    
    if 'image' in Config.input_feature and Config.multi_process == 1:
        print("\nStarting image preprocessing")
        image_preprocess(Config)


def match_files(Config):
    data_folder_list = os.listdir(Config.input_features)
    
    no_match = {}
    match = []
    
    total_files = {}
    
    max_file_number = 0
    for feature in Config.input_features:
        try:
            temp_list = [name.split('.')[0] for name in os.listdir(os.path.join(Config.data_path, feature))]
        except
        total_files[feature] = temp_list
        max_file_number = max(max_file_number, len(temp_list))
    
    lists = list(total_files.values())
    
    # find common files
    if lists:
        match = set(lists[0])
        for lst in lists[1:]:
            match.intersection_update(lst)
    else:
        match = set()
    
    all_strings = set(item for lst in lists for item in lst)
    for key, lst in total_files.items():
        no_match[key] = list(set(lst) - (all_strings - set(lst)))
    

if __name__ == "__main__":
    class test_config:
        data_path = "D:/home/BCML/IITP-multimodal/data/test_data/"
        input_features = ['eeg', 'rppg', 'ecg', 'video']
        
    config = test_config
    match_files(config)
    