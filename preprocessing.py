import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Manager
from utils.preprocessing.visual_data_preprocessing import get_video_creation_time
from utils.preprocessing.physiological_signal_preprocessing import get_ECGnPPG_creation_time

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

def preprocess(Config):
    
    if not isinstance(Config.data_path, str):
        raise TypeError(f"data path: {Config.data_path} is not a string")
    
    if not isinstance(Config.cache_path, str):
        raise TypeError(f"save path: {Config.save_path} is not a string")
    
    match_files(Config)
    
    data_types = os.listdir(Config.data_path)
    
    if 'video' in Config.input_features and Config.multi_process == 1:
        print("\nStarting video preprocessing")
        video_preprocess(Config)
    elif 'video' in Config.input_features and Config.multi_process >= 2:
        print("\nStarting video preprocessing with multiprocess")
    
    if 'image' in Config.input_feature and Config.multi_process == 1:
        print("\nStarting image preprocessing")
        image_preprocess(Config)

def match_files(Config):
    return 0
             

   