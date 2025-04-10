import os
from config import Preprocessing_Config
from preprocessing import preprocess

def get_preprocess_config():
    """
    Returns
    -------
    dataclass
        Refer the 'config.py' file.
    """
    return Preprocessing_Config().video_and_rppg()

if __name__ == "__main__":
    preprocess_config = get_preprocess_config
    if preprocess_config.preprocess:
        preprocess(preprocess_config)
        
    