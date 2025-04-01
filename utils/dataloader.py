# core libraries
import os
import sys
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt

# torch libraries
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

# video libraries
import cv2

# string library
import re

# mat library
import scipy.io as sio

def split_ecg():
    base_path = "D:/home/BCML/IITP/data/16channel_Emotion/preprocessing_data/ecg/filtered_ecg_signals.csv"
    file = pd.read_csv(base_path, index_col=0)
    for index in file.index.unique().tolist():
        select_index = np.array(deepcopy(file.loc[index]))
        select_index = select_index[:, 1]
        np.save(f"D:/home/BCML/IITP/data/16channel_Emotion/preprocessing_data/ecg/ecg_{index}.npy", select_index)


class MultiModalDataset(Dataset):
    """
    A PyTorch Dataset for handling multi-modal data: video, time-series, and text.

    Args:
    """
    
    def __init__(self, data_dir, time_window=1):

    def __len__(self):

    def __getitem__(self, idx):

    
    def cal_total_data_length(self):
        total_window_count_list = []
        
        for eeg_file in self.eeg_list:
            time_ground_truth = sio.loadmat(os.path.join(self.eeg_dir, eeg_file))
            total_window_count_list.append(int(int(len(time_ground_truth['emotion_label'][0])/self.eeg_rate)/self.time_window))
            
        return total_window_count_list
    
    def set_files(self, count):
        
        return input_x, input_y

    

if __name__ == "__main__":
    base_path = "C:/Users/U/Desktop/BCML/IITP/IITP_2025/data/16channel_Emotion/test_data/"