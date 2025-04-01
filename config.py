import os
import sys
from dataclasses import dataclass, asdict, field

""" 
Data feature notation convention
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
        data_chunk (float): The data slicing window (seconds)
                            The minimum time window is 0.2 second.
                            It should be 0.2 * natural number
                             
                       | Sampling rate | minimum slicing |
            EEG, label |     600 Hz    |     120 idx     |
            ECG        |     130 Hz    |      26 idx     |
            PPG        |     135 Hz    |      27 idx     |
            video      |      30 Hz    |       6 frames  |
    """
    
    """ Fixed by the sampling rates of each sensors """
    EEG_label = 120
    ECG = 26
    PPG = 27
    video = 6
    
    """ The customizable variables """
    preprocess: bool = True
    data_path: str = "./data/test_data"  # this path should be changed
    cache_path: str = "./data/preprocessed_data"  # this path should be changed
    input_features: list[str] = field(default_factory=list)
    multi_process: int = 1  # multiprocessding process number
    face_detector: str = "mediapipe"
    default_crop_size: list[int, int] = field(default_factory=list)  # 
    resize_ratio: float = 1.0
    data_chunk: float = 0.2
    
    @classmethod
    def pilot_test(cls):
        return cls(input_features=['eeg', 'rppg', 'ecg', 'video'],
                   default_crop_size=[640, 640])
    
    @classmethod
    def only_time_series(cls):
        return cls(input_features=['eeg', 'rppg', 'ecg'],
                   default_crop_size=[640, 640])
    
    @classmethod
    def only_video(cls):
        return cls(input_features=['video'],
                   default_crop_size=[640, 640])
    
    def get_config(exp_name="pilot_test"):
        return getattr(Config, exp_name)()
    
    