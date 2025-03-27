import os
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Manager

def preprocess(Config):
    data_types = os.listdir(Config.data_path)
    
    if 'video' in Config.input_features:
        print("\nStart video preprocessing")
        video_preprocess(os.path.join(Config.data_path, 'video'), os.path.join(Config.cache_path, "video"), Config.resize_ratio)
        
    if 'image' in Config.input_feature:
        print("\nStart image preprocessing")
        image_preprocess(os.path.join(Config.data_path, 'image'), os.path.join(Config.cache_path, "image"), Config.resize_ratio)

def video_preprocess(data_path, save_path, resize_ratio):
    video_list = os.listdir(data_path)
    
    total_frames = 0
    max_frame = 0
    frame_num_list = []
    for video in video_list:
        cap = cv2.VideoCapture(os.path.join(data_path, video))
        if not cap.isOpened():
            print(f"\n*** Failed to open the video: {video} ***")
            continue
        number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames += number_of_frames
        max_frame = max(max_frame, number_of_frames)
        frame_num_list.append(number_of_frames)
    
    print(f"Total frames to process: {total_frames}")
    frame_start_num = (max_frame // 10 + 1) * 10
    
    with tqdm(total=total_frames, desc="Preprocessing all videos with 1 process", unit='frame') as pbar:
        for i, video in enumerate(video_list):
            frame_count = frame_start_num
            cap = cv2.VideoCapture(os.path.join(data_path, video))
            if not cap.isOpened():
                print(f"\n*** Failed to open the video: {video} ***")
                continue
            
            print("Processing video: {video} ({frame_num_list[i]} frames)")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                crop_frame = face_crop(frame)
                if resize_ratio != 1.0:
                    crop_frame = cv2.resize(crop_frame, (0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
                save_path = os.path.join(save_path, f"{frame_count}.png")
                cv2.imwrite(save_path, crop_frame)
                pbar.update(1)
                frame_count += 1
            cap.release()
        print("\nFinished preprocessing all videos.")
    
    
def video_multi_preprocess(data_path, save_path):
    video_list = os.listdir(data_path)
    
    max_frame = 0
    for video in video_list:
        cap = cv2.VideoCapture(os.path.join(data_path, video))
        if not cap.isOpened():
            print(f"\n*** Failed to open the video: {video} ***")
            continue
        
        max_frame = max(max_frame, number_of_frames)
    
    frame_start_num = (max_frame // 10 + 1) * 10
    
    
    
def face_crop(image):
    return image

""" The codes taken from the 'BaseLoader.py' file in the library rppg-toolbox """

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
                
if __name__ == "__main__":
    """ Change directories to run the preprocess function. """
    data_path = "D:\home\BCML\IITP-multimodal\data\test_data\video"
    save_path = 
    resize_ratio = 
    video_preprocess()