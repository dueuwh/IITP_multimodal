import os
import numpy as np
import scipy.io as sio
import pandas as pd

def get_ECGnPPG_creation_time():
    return 0

def synchronization_UNIXTIME():
    return 0    


""" The following code is taken from the 'BaseLoader.py' file in the rppg-toolbox library """

class math_tools:
    @staticmethod
    def diff_normalize_data(data):
        """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
        n, h, w, c = data.shape
        diffnormalized_len = n - 1
        diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
        diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
        for j in range(diffnormalized_len):
            diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                    data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
        diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
        diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
        diffnormalized_data[np.isnan(diffnormalized_data)] = 0
        return diffnormalized_data
    
    @staticmethod
    def diff_normalize_label(label):
        """Calculate discrete difference in labels along the time-axis and normalize by its standard deviation."""
        diff_label = np.diff(label, axis=0)
        diffnormalized_label = diff_label / np.std(diff_label)
        diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
        diffnormalized_label[np.isnan(diffnormalized_label)] = 0
        return diffnormalized_label
    
    @staticmethod
    def standardized_data(data):
        """Z-score standardization for video data."""
        data = data - np.mean(data)
        data = data / np.std(data)
        data[np.isnan(data)] = 0
        return data
    
    @staticmethod
    def standardized_label(label):
        """Z-score standardization for label signal."""
        label = label - np.mean(label)
        label = label / np.std(label)
        label[np.isnan(label)] = 0
        return label
    
    @staticmethod
    def resample_ppg(input_signal, target_length):
        """Samples a PPG sequence into specific length."""
        return np.interp(
            np.linspace(
                1, input_signal.shape[0], target_length), np.linspace(
                1, input_signal.shape[0], input_signal.shape[0]), input_signal)