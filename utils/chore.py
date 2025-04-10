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
import cv2

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
    
    
    # start_time_list = []
    # path = "D:/home/BCML/IITP-multimodal/data/test_data/video/"
    # video_list = os.listdir(path)
    # for video in video_list:
    #     print(os.path.join(path, video))
    #     start_time_list.append(get_video_creation_time_mediainfo(os.path.join(path, video)))
    
    
    def analyze_strings(data):
        """
        Analyze strings in a dictionary where each key is associated with a list of strings.
        This function identifies:
        1. Common strings present in all lists.
        2. Strings unique to each key's list.
    
        Args:
            data (dict): A dictionary where keys are associated with lists of strings.
    
        Returns:
            dict: A dictionary with two keys:
                - 'common': A list of strings present in all lists.
                - 'unique': A dictionary where each key corresponds to the original dictionary keys,
                  and the value is a list of strings unique to that key's list.
        """
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary.")
    
        # Extract all the lists from the dictionary
        lists = list(data.values())
        
        # Check if all values are lists
        if not all(isinstance(lst, list) for lst in lists):
            raise ValueError("All values in the dictionary must be lists.")
        
        # Find common strings present in all lists
        if lists:
            common_strings = set(lists[0])
            for lst in lists[1:]:
                common_strings.intersection_update(lst)
        else:
            common_strings = set()
        
        # Find unique strings for each key's list
        unique_strings = {}
        all_strings = set(item for lst in lists for item in lst)
        for key, lst in data.items():
            unique_strings[key] = list(set(lst) - (all_strings - set(lst)))
        
        return {
            'common': list(common_strings),
            'unique': unique_strings
        }

    # Example usage
    test_input = {'eeg': ['160_1', '160_2'], 'video': ['160_1']}
    result = analyze_strings(test_input)
    print("Common Strings:", result['common'])
    print("Unique Strings:", result['unique'])