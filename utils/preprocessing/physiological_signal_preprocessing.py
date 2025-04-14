import os
import numpy as np
import scipy.io as sio
import pandas as pd
from datetime import datetime

#  dev libraries
import matplotlib.pyplot as plt
import sys


def get_physiological_creation_time(Config):
    """
    Parameters
    ----------
    Config : dataclass
        Refer the 'config.py' file.

    Returns
    -------
    creation_time : dict
        The creation time for all files.

    """
    creation_time = {}
    
    for feature in Config.input_features:
        if feature == 'video':
            pass
        else:
            creation_time[feature] = {}
            for file in Config.match_files:
                creation_time[feature][file] = get_file_creation_time(os.path.join(Config.data_path, f"{feature}/{file}{Config.extensions[feature]}"))
    return creation_time


def get_file_creation_time(file_path):
    """
    Parameters
    ----------
    file_path : str
        literally file path.

    Raises
    ------
    FileNotFoundError
        It is not a file path (it is a folder path or something else.)
    ValueError
        The extension is not supported.
    OSError
        Cannot find the creation time.

    Returns
    -------
    readable_time : list
        The creation time of each files.

    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    
    if not (file_path.endswith(".csv") or file_path.endswith(".npy") or file_path.endswith(".mat")):
        raise ValueError(f"The file '{file_path}' is not a .csv or .npy or .mat file.")
    
    try:
        creation_time = os.path.getctime(file_path)
        readable_time = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
        return readable_time
    except Exception as e:
        raise OSError(f"Error retrieving the creation time of the file: {e}")


def index2time(data, selected_feature, Config):
    total_time = {}
    for file in Config.match_files:
        total_time[file] = round(len(data[file])//Config.minimum_chunk[selected_feature]*Config.data_chunk, 1)
    return total_time


class lib_preprocess:  #  label index base preprocess
    def __init__(self, feature_others, save_num_iter, Config, eeg_remain_time):
        self.feature_others = feature_others
        self.save_num_iter = save_num_iter
        self.Config = Config
        self.eeg_remain_time = eeg_remain_time
        
        self.save_file_size_ecg = int(Config.save_file_size * Config.minimum_chunk['ecg'] * (1//Config.data_chunk))
        self.save_file_size_ppg = int(Config.save_file_size * Config.minimum_chunk['ppg'] * (1//Config.data_chunk))
        self.save_file_size_rppg = int(Config.save_file_size * Config.minimum_chunk['rppg'] * (1//Config.data_chunk))
        self.save_file_sizes = {'ecg':self.save_file_size_ecg, 'ppg':self.save_file_size_ppg, 'rppg':self.save_file_size_rppg}
        
        self.ecg_remain = int(eeg_remain_time//Config.sampling_rate['ecg'])
        self.ppg_remain = int(eeg_remain_time//Config.sampling_rate['ppg'])
        self.rppg_remain = int(eeg_remain_time//Config.sampling_rate['rppg']) 
        self.save_file_remains = {'ecg':self.ecg_remain, 'ppg':self.ppg_remain, 'rppg':self.rppg_remain}
        
        for feature in feature_others:
            os.makedirs(os.path.join(self.Config.cache_path, feature), exist_ok=True)
    
    def __save(self, feature, data):
        for file in self.Config.match_files:
            current_index = 0
            for i in range(self.save_num_iter):
                np.save(os.path.join(self.Config.cache_path, f"{feature}/{file}_{i}.npy"), data[current_index : current_index + self.save_file_sizes[feature]])
                current_index += self.save_file_sizes[feature]
            np.save(os.path.join(self.Config.cache_path, f"{feature}/{file}_{i+1}.npy"), data[current_index : current_index + self.save_file_remains[feature]])
    
    def save(self, feature, file_name):
        if feature == 'ecg':
            temp_load = np.array(pd.read_csv(os.path.join(self.Config.data_path, f"ecg/{file_name}.csv"))[['ECG']])
            self.__save('ecg', temp_load)
        elif feature == 'ppg':
            temp_load = np.array(pd.read_csv(os.path.join(self.Config.data_path, f"ppg/{file_name}.csv"))[['PPG']])
            self.__save('ppg', temp_load)
        elif feature == 'rppg':
            temp_load = np.load(os.path.join(self.Config.data_path, f"rppg/{file_name}.npy"))
            self.__save('rppg', temp_load)


def label_index_base_preprocessing(Config):
    """

    Parameters
    ----------
    Config : TYPE
        Refer the "config.py' file.

    Returns
    -------
    None.
    
    Purpose
    -------
    Synchronization of physiological data.
    This function works well only when the eeg data is included in the "input_features" parameter of Config.
    In this method, the eeg files must all be preprocessed (All files must have same shape).
    Refer the preprocessed multimodal dataset (160 files)
    """
    
    if 'eeg' not in Config.input_features:
        raise ValueError("Label_Time_Base type preprocess works well only when the eeg data is included in the 'input_features' parameter of Config.")
    
    eeg = {}
    emotion_label = {}
    for file in Config.match_files:
        temp_load = sio.loadmat(os.path.join(Config.data_path, f"eeg/{file}"))
        
        #  For key variation, search for 'eeg' and 'label' keyword in key.
        eeg[file] = [value for key, value in temp_load.items() if 'eeg' in key][0]
        emotion_label[file] = [value[0] for key, value in temp_load.items() if 'label' in key][0]

    eeg_file_length = eeg[file].shape[1]
    eeg_time = round(eeg_file_length//Config.minimum_chunk['eeg']*Config.data_chunk, 1)
    eeg_remain_time = 0
    
    os.makedirs(os.path.join(Config.cache_path, "eeg"), exist_ok=True)
    save_file_size_index = int(Config.save_file_size * Config.minimum_chunk['eeg'] * (1//Config.data_chunk))
    eeg_save_num_iter = eeg_file_length//save_file_size_index
    for file_index, file in enumerate(Config.match_files):
        current_index = 0
        for i in range(eeg_save_num_iter):
            np.save(os.path.join(Config.cache_path, f"eeg/{file}_{i}.npy"), eeg[file][:, current_index : current_index + save_file_size_index])
            current_index += save_file_size_index
        np.save(os.path.join(Config.cache_path, f"eeg/{file}_{i+1}.npy"), eeg[file][:, current_index:])
        if file_index == 0:
            eeg_remain_time = round(len(eeg[file][:, current_index:])/Config.sampling_rate['eeg'], 1)

    if len(Config.input_features) >= 2:
        feature_others = [feature for feature in Config.input_features if feature != 'eeg']
        
        lib_others = lib_preprocess(feature_others, eeg_save_num_iter, Config, eeg_remain_time)
        
        for feature in feature_others:
            for file in Config.match_files:
                lib_others.save(feature, file)
                
    Config.eeg_time = eeg_time
    Config.eeg_remain_time = eeg_remain_time
    
def creation_time_base(Config):
    return 


def check_chunk(data):
    return data


def interpolation_by_chunk():
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