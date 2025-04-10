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
        preprocess (bool) : Whether to run preprocess or not, if run first time, this paramether should be True.
        data_path (str) : The raw data path (The video should be trimmed before preprocess).
        cache_path (str) : The path to save preprocessed data.
        input_features (list[str, str, ..., str]) : Input features.
        multi_process (bool) : Whether to run preprocess in multiprocess or not.
        face_detector: (str) : The "mediapipe" is the only method implemented so far.
        default_crop_size (list[int, int]) : Height, width. This is the input size if resize_ratio equals to 1.0
        resize_ratio (float) : The resize ratio for cropped frame, decrease if out of memory error occurs
        resize_pixel (list[int, int]) : The resize pixel size for cropped frame.
        data_chunk (float) : The data slicing window (seconds)
                            The minimum time window is 0.2 second.
                            It should be 0.2 * natural number
        match_files (list[str, str, ...]) : Common file names exist in all feature folders
        extensions (dict) : The extension for each physiological raw data files.
        dataset (str) : dataset name
                        -name list
                         "16channel_kw" : The 16channel experiments conducted in kwangwoon university
                         "DEAP" : DEAP dataset
                         "16channel_Yonsei" : The 16channel experiments conducted in yonsei university
                                              * This dataset is still a work in progress - 2025.04.10
                        
                       | Sampling rate | minimum slicing |
            EEG, label |     600 Hz    |     120 idx     |
            ECG        |     130 Hz    |      26 idx     |
            PPG        |     135 Hz    |      27 idx     |
            video      |      30 Hz    |       6 frames  |
    """
    
    """ Fixed by the sampling rates of each sensors """
    eeg_label = 120
    ecg = 26
    ppg = 27
    video = 6
    
    """ The customizable variables """
    preprocess: bool = True
    data_path: str = "./data/test_data"  # this path should be changed
    cache_path: str = "./data/preprocessed_data"  # this path should be changed
    input_features: list[str] = field(default_factory=list)
    multi_process: int = 1  # multiprocessding process number
    face_detector: str = "mediapipe"
    default_crop_size: list[int, int] = field(default_factory=list)
    resize_ratio: float = 1.0
    resize_pixel: list[int, int] = field(default_factory=list)
    data_chunk: float = 0.2
    match_files: list[str] = field(default_factory=list)
    extensions: dict = field(default_factory=dict)
    dataset: str = "16channel_kw"
    
    @classmethod
    def pilot_test(cls):
        return cls(input_features=['eeg', 'rppg', 'ecg', 'video'],
                   default_crop_size=[640, 640],
                   extensions={'eeg':'.mat', 'ecg':'.csv', 'rppg':'.npy', 'video':'.mp4'})
    
    @classmethod
    def only_time_series(cls):
        return cls(input_features=['eeg', 'rppg', 'ecg'],
                   default_crop_size=[640, 640],
                   extensions={'eeg':'.mat', 'ecg':'.csv', 'rppg':'.npy', 'video':'.mp4'})
    
    @classmethod
    def only_video(cls):
        return cls(input_features=['video'],
                   default_crop_size=[640, 640],
                   extensions={'eeg':'.mat', 'ecg':'.csv', 'rppg':'.npy', 'video':'.mp4'})
    
    @classmethod
    def video_and_rppg(cls):
        return cls(input_features=['video'],
                   default_crop_size=[640, 640],
                   resize_pixel=[48, 48],
                   extensions={'eeg':'.mat', 'ecg':'.csv', 'rppg':'.npy', 'video':'.mp4'})
    
    def get_config(exp_name="pilot_test"):
        return getattr(Preprocessing_Config, exp_name)()

@dataclass
class Model_Config:
    """
    example
    """
    model_name = "sota"
    
    @classmethod
    def pilot_test(cls):
        return cls(model_name="sota")
    