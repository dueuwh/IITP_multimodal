import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from dataclasses import dataclass
from typing import List
import codecs
from pymediainfo import MediaInfo

def get_video_creation_time_mediainfo(video_path):
    """
    pymediainfo를 사용하여 영상 파일의 촬영 시작 시간(creation_time)을 가져옵니다.

    Args:
        video_path (str): 영상 파일 경로.

    Returns:
        str: 촬영 시작 시간 또는 None.
    """
    media_info = MediaInfo.parse(video_path)

    for track in media_info.tracks:
        if track.track_type == "General":
            return track.tagged_date  # 촬영 시작 시간

    return None

def remove_unnecessary_counter(path):
    files = os.listdir(path)
    for file in tqdm(files):
        file_dir = os.path.join(path, file)
        if not os.path.isfile(file_dir):
            continue
        
        if '-1' in file:
            new_name = file[:file.index('-1')]
        else:
            new_name = file
        
        new_file_dir = os.path.join(path, f"{new_name}.csv")
        
        os.rename(file_dir, new_file_dir)

# @dataclass
# class C:
#     x: List = []
#     def add(self, element):
#         self.x += element

def replace_backslash_with_slash(input_string):
    """
    Replace all backslashes in the input string with forward slashes.

    Args:
        input_string (str): The input string containing backslashes.

    Returns:
        str: A string with all backslashes replaced by forward slashes.
    """
    return input_string.replace("\\", "/")


if __name__ == "__main__":
    # path = "D:/home/BCML/IITP/data/16channel_Emotion/Polar/"
    # remove_unnecessary_counter(path)
    
    # o1 = C()
    # o2 = C()
    # o1.add(1)
    # o2.add(2)
    # assert o1.x == [1, 2]
    # assert o1.x is not o2.x
    # max_number = 0
    # for i in range(100):
    #     max_number = max(max_number, np.random.randint(0, 100))
    # print(max_number)
    
    a = "C:\home\data\test_file.py"
    
    
    start_time_list = []
    path = "D:/home/BCML/IITP-multimodal/data/test_data/video/"
    video_list = os.listdir(path)
    for video in video_list:
        print(os.path.join(path, video))
        start_time_list.append(get_video_creation_time_mediainfo(os.path.join(path, video)))
    