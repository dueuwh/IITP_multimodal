import os
import numpy as np
from tqdm import tqdm
from utils.preprocessing.visual_data_preprocessing import get_video_creation_time, video_preprocess, video_multi_preprocess, image_preprocess, image_multi_preprocess
from utils.preprocessing.physiological_signal_preprocessing import get_physiological_creation_time, label_index_base_preprocessing
from utils.preprocessing.physiological_signal_preprocessing import creation_time_base_preprocessing

#  dev
import sys

def preprocess(Config):
    
    """
        Data folder structure should be:
        ****** All folder names must be in lower case and singular notation ******
        |---root
            |---video (or image)
                
                |---160_1.mp4
                    160_2.mp4
                    ...
                    16n_n.mp4
                    
            |---image (The images should simply be extracted frames from the video)
                |---160_1
                    |---000.png
                        001.png
                        ...
                        nnn.png
                |---160_2
                    |---000.png
                        001.png
                        ...
                        nnn.png
                    
                    ...
                    
                |---16n_n
                |---000.png
                        001.png
                        ...
                        nnn.png
                    
            |---eeg
                |---160_1.mat
                    160_2.mat
                    ...
                    16n_n.mat
                    
            |---eeg_raw
                |---000  (subject number)
                    |---1  (experiment number)
                        |---160_1_07_01_2025_14_13_10_0000.mat (raw file name)
                            160_1_07_01_2025_14_13_10_0001.mat
                            ...
                            160_1_07_01_2025_14_13_10_0047.mat
                    |---...
                    |---n
                        |---...
                |---00n
                    |---...
                    
            |---polar (ecg)
                |---160_1.csv
                    160_2.csv
                    ...
                    16n_n.csv
                    
            |---rppg (if exist)
                |---160_1.npy
                    160_2.npy
                    ...
                    16n_n.npy
    """
    
    if not isinstance(Config.data_path, str):
        raise TypeError(f"data path: {Config.data_path} is not a string")
    
    if not isinstance(Config.cache_path, str):
        raise TypeError(f"save path: {Config.save_path} is not a string")
    
    if Config.preprocess_type == "Creation_Time_Base":
        ctb_inst = creation_time_base_preprocessing(Config)
        ctb_inst.eeg_raw_preprocess()
    
    match, no_match = match_files(Config)
    print(f"{'='*50}")
    print("Check file names\n")
    print(f"Common files: {match}")
    if len(no_match) == 0:
        print("No unique files")
    else:
        for key, value in no_match.items():
            print(f"Unique files: {key} : {value}")
        print("Unique files will not be preprocessed")
    print(f"{'='*50}")
    
    Config.match_files = match
    
    if Config.preprocess_type == "Creation_Time_Base":
        # video_creation_times = get_video_creation_time(Config)
        # physigological_creation_times = get_physiological_creation_time(Config)
        
        creation_time_base(Config)
        
    elif Config.preprocess_type == "Label_Time_Base":
        label_index_base_preprocessing(Config)

        if 'video' in Config.input_features and Config.multi_process == 1:
            print("\nStarting video preprocessing")
            video_preprocess(Config)
        elif 'video' in Config.input_features and Config.multi_process >= 2:
            print("\nStarting video preprocessing with multiprocess")
    
    else:
        raise ValueError(f"{Config.preprocess_type} is not surpported")
    
    # if 'image' in Config.input_feature and Config.multi_process == 1:
    #     print("\nStarting image preprocessing")
    #     image_preprocess(Config)


def match_files(Config):
    """
    Parameters
    ----------
    Config : dataclass
        Refer the 'config.py' file.

    Returns
    -------
    match : list(str, str, ...)
        Common file names exist in all feature folders.
    no_match : dict(str:list(str, str, ...), ...)
        Unique file names exist each feature folders.

    """
    
    no_match = {}
    match = []
    
    total_files = {}
    
    max_file_number = 0
    for feature in Config.input_features:
        temp_list = [name.split('_')[0]+'_'+name.split('_')[1].split('.')[0] for name in os.listdir(os.path.join(Config.data_path, feature))]
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
    
    # find unique files
    all_strings = set(item for lst in lists for item in lst)
    for key, lst in total_files.items():
        unique_list = list(set(lst) - match)
        if len(unique_list) == 0:
            pass
        else:
            no_match[key] = unique_list
    
    return list(match), no_match

if __name__ == "__main__":
    print("nope")