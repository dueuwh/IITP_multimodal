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
    """preprocess arguments"""
    
    preprocess: bool = True
    data_path: str = "./data/test_data"  # this path should be changed
    cache_path: str = "./data/preprocessed_data"  # this path should be changed
    input_features: list[str] = field(default_factory=list)
    multi_process: int = 1  # multiprocessding process number
    face_detector: str = "mediapipe"
    resize_ratio: float = 1.0  # resize ratio for cropped frame
    
    @classmethod
    def pilot_test(cls):
        return cls(input_features=['eeg', 'rppg', 'ecg', 'video'])
    
    def get_config(exp_name="pilot_test"):
        return getattr(Config, exp_name)()
    
    