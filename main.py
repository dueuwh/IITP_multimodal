import os
from config import Preprocessing_Config
from preprocessing import preprocess

if __name__ == "__main__":
    if Preprocessing_Config.preprocess:
        preprocess(Preprocessing_Config.pilot_test())
        
    