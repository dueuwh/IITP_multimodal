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
        video_data (list): A list of video tensors (e.g., each tensor shape: [frames, height, width, channels]).
        time_series_data (list): A list of time-series tensors (e.g., each tensor shape: [sequence_length, features]).
        text_data (list): A list of text strings.
        labels (list): A list of labels corresponding to the data.
        text_tokenizer (callable): A function or callable to tokenize text into tensors.
     
    Data folder structure should be:
    ****** All folder names must be in lower case and singular notation ******
        root
            video (or image)
            
                160_1.mp4
                160_2.mp4
                ...
                16n_n.mp4
                
            image
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
    
    @staticmethod
    def check_file_names(directory_list):
        return 0
    
    @staticmethod
    def data_visualization(data):
        plt.plot(data)
        plt.show()
    
    def __init__(self, data_dir, time_window=1):
        
        """
        Parameters
            data_dir (str)   : data directory
            time_window (float): data slicing window (second)
                                 minimum time window is 0.2 second
                                 should be 0.2 * natural number
                                 
                           | Sampling rate | minimum slicing |
                EEG, label |     600 Hz    |     120 idx     |
                ECG        |     130 Hz    |      26 idx     |
                PPG        |     135 Hz    |      27 idx     |
                video      |      30 Hz    |       6 idx     |
        """
        
        assert(time_window % 0.2 == 0), "parameter 'time_window' is not an integer multiple of 0.2"
        
        self.eeg_rate = 600
        self.ecg_rate = 130
        self.ppg_rate = 135
        self.video_rate = 30
        
        self.time_window = time_window
        self.data_dir = data_dir
        data_types = [folder for folder in os.listdir(self.data_dir) if not os.path.isfile(self.data_dir)]
        
        for folder_name in data_types:
            if folder_name.lower() == 'video':
                self.video_dir = os.path.join(self.data_dir, folder_name)
                self.video_list = os.listdir(self.video_dir)
            elif folder_name.lower() == 'ecg':
                self.ecg_dir = os.path.join(self.data_dir, folder_name)
                self.ecg_list = os.listdir(self.ecg_dir)
            elif folder_name.lower() == 'rppg':
                self.rppg_dir = os.path.join(self.data_dir, folder_name)
                self.rppg_list = os.listdir(self.rppg_dir)
            elif folder_name.lower() == 'eeg':
                self.eeg_dir = os.path.join(self.data_dir, folder_name)
                self.eeg_list = os.listdir(self.eeg_dir)
            else:
                print("* Warning! folder unrelated with dataset is found: ", folder_name)
        
        assert(len(self.video_list) == len(self.ecg_list) == len(self.rppg_list) == len(self.eeg_list))
        
        self.file_number = len(self.eeg_list)
        
        # calculate total data length
        self.total_window_count_list = self.cal_total_data_length()
        
        # self.input_x, self.input_y = 
        

    def __len__(self):
        return sum(self.total_window_count_list)

    def __getitem__(self, idx):
                
        
        
        # label_tensor = torch.tensor(label, dtype=torch.float32)

        return 
    
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
    data_types = os.listdir(base_path)
    dataset_test = MultiModalDataset(base_path)
    print(dataset_test.total_window_count_list)