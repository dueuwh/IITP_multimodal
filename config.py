import os
import sys
from dataclasses import dataclass, asdict, field

""" 
Data feature notation convention in folder name and this file
1. lower
2. sigular
"""

@dataclass
class Preprocessing_Config:
    """
    Parameters
        minimum_chunk (dict) : The minimum data length in a time chunk (0.2 second, refer the below table)
        preprocess (bool) : Whether to run preprocess or not, if run first time, this paramether should be True.
        preprocess_type (str) : The preprocessing type.
                                "Label_Time_Base" for handmade preprocessing. 
                                *** This method works only when the eeg feature is included in input_features
                                "Creation_Time_Base" for auto synchronization based on creation time of file.
        seperate_label (bool) : Whether to seperate the data based on label or not.
        data_path (str) : The raw data path (The video should be trimmed before preprocess).
        cache_path (str) : The path to save preprocessed data.
        input_features (list[str, str, ..., str]) : Input features.
        multi_process (bool) : Whether to run preprocess in multiprocess or not.
        face_detector: (str) : The "mediapipe" is the only method implemented so far.
        default_crop_size (list[int, int]) : Height, width. This is the input size if resize_ratio equals to 1.0
        resize_ratio (float) : The resize ratio for cropped frame, decrease if out of memory error occurs
        resize_pixel (tuple[int, int]) : Height, width. The resize pixel size for cropped frame.
                                        If the pixel size is 1, it regards as "Do not resize by referencing this parameter"
        data_chunk (float) : The minimum time window. Fixed value.
        match_files (list[str, str, ...]) : Common file names exist in all feature folders
        extensions (dict) : The extension for each physiological raw data files.
        dataset (str) : dataset name
                        -name list
                         "16channel_kw" : The 16channel experiments conducted in kwangwoon university include 2nd dataset (with ppg)
                         "DEAP" : DEAP dataset
                         "16channel_Yonsei" : The 16channel experiments conducted in yonsei university
                                              * This dataset is still a work in progress - 2025.04.10
        save_file_size (int) : The preprocessed file size (second).
                               ex) one file size = (save_file_size, int(eeg data length(index) // 120), 120)
        
                       | Sampling rate | minimum slicing |
            EEG, label |     600 Hz    |     120 idx     |
            ECG        |     130 Hz    |      26 idx     |
            PPG        |     135 Hz    |      27 idx     |
            video      |      30 Hz    |       6 frames  |
            rppg       |      30 Hz    |       6 frames  | same as video
    """
    
    """ Fixed by the sampling rates of each sensors """
    minimum_chunk: dict = field(default_factory=lambda: {"eeg":120, "ecg":26, "ppg":27, "rppg":6, "video":6})
    sampling_rate: dict = field(default_factory=lambda: {"eeg":600, "ecg":130, "ppg":135, "rppg":30, "video":30})
    
    """ The customizable variables """
    preprocess: bool = True
    preprocess_type: str = "Label_Time_Base"  # or "Creation_Time_Base"
    spearate_label: bool = False
    data_path: str = f"{os.path.dirname(os.path.abspath(__file__))}/data/test_data"  # this path should be changed
    cache_path: str = f"{os.path.dirname(os.path.abspath(__file__))}/data/preprocessed_data"  # this path should be changed
    input_features: list[str] = field(default_factory=list)
    multi_process: int = 1  # multiprocessding process number
    face_detector: str = "mediapipe"
    default_crop_size: list[int, int] = field(default_factory=list)
    resize_ratio: float = 1.0
    resize_pixel: tuple[int, int] = field(default_factory=lambda: (1, 1))
    data_chunk: float = 0.2
    match_files: list[str] = field(default_factory=list)
    extensions: dict = field(default_factory=lambda: {'eeg': '.mat', 'ecg': '.csv', 'rppg': '.npy', 'video': '.mp4'})
    dataset: str = "16channel_kw"
    save_file_size: int = 60
    
    @classmethod
    def pilot_test(cls):
        return cls(input_features=['eeg', 'ecg', 'video'],
                   default_crop_size=[640, 640])
    
    @classmethod
    def MBE(cls):  # Medical & Bioogical Engineering - 2025 spring
        return cls(input_features=['video', 'eeg', 'ecg'],
                   default_crop_size=[640, 640],
                   resize_pixel=[48, 48],
                   preprocess_type="Label_Time_Base",
                   data_path=f"{os.path.dirname(os.path.abspath(__file__))}/data/BME_2025_spring",
                   cache_path=f"{os.path.dirname(os.path.abspath(__file__))}/data/BME_2025_spring_preprocessed_data")
    
    @classmethod
    def only_time_series(cls):
        return cls(input_features=['eeg', 'rppg', 'ecg'],
                   default_crop_size=[640, 640],)
    
    @classmethod
    def only_video(cls):
        return cls(input_features=['video'],
                   default_crop_size=[640, 640],)
    
    @classmethod
    def video_and_rppg(cls):
        return cls(input_features=['video'],
                   default_crop_size=[640, 640],
                   resize_pixel=[48, 48],)
     
    def get_config(exp_name="pilot_test"):
        return getattr(Preprocessing_Config, exp_name)()


@dataclass
class Model_Config:
    model_name = "BLIP"
    image_encoder: str = "FER_1"
    visual_2d_positional_encoder: str = "none"
    physiological_positional_encoder: str = "naive"
    
    @classmethod
    def blip_default(cls):
        return cls()
    

@dataclass
class Train_Config:    
    learning_rate: float = 1e-5
    epoch: int = 100
    early_stopping: bool = True
    optimizer: str = "adam"
    
    
    @classmethod
    def blip_train_1(cls):
        return cls()

