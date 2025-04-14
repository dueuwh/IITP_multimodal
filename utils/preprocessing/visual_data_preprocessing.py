import os
import numpy as np
import cv2
from tqdm import tqdm
import mediapipe as mp
from pymediainfo import MediaInfo
from multiprocessing import Process, Manager

def get_video_creation_time(Config):
    video_time_list = {}
    for video in Config.match_files:
        media_info = MediaInfo.parse(os.path.join(Config.data_path, f"video/{video}{Config.extensions['video']}"))
        for track in media_info.tracks:
            if track.track_type == "General":
                video_time_list[video] = track.tagged_date.split(' ')[0]+' '+track.tagged_date.split(' ')[1]
                
    return video_time_list


def image_preprocess(Config):
    """
    Parameters
    ----------
    Config : dataclass
        Refer the 'config.py' file.

    Returns
    -------
    Nothing

    Purpose
    -------
    Crop a face from a image and save it as npy files.
    
    """


def image_multi_preprocess(Config):
    return 0


def video_preprocess(Config):
    """
    Parameters
    ----------
    Config : dataclass
        Refer the 'config.py' file.

    Returns
    -------
    Nothing

    Purpose
    -------
    Crop a face from a video and save it as npy files.
    
    """

    data_path = os.path.join(Config.data_path, "video")
    save_path = os.path.join(Config.cache_path, "video")
    crop_size = Config.default_crop_size
    resize_ratio = Config.resize_ratio
    resize_pixel = Config.resize_pixel
    frame_chunk = int(Config.data_chunk*Config.minimum_chunk["video"])
    
    save_frame_h, save_frame_w = crop_size
    
    if resize_ratio != 1.0:
        save_frame_h = int(save_frame_h * resize_ratio)
        save_frame_w = int(save_frame_w * resize_ratio)
    
    if resize_pixel[0] != 1 and resize_pixel[1] != 1:
        save_frame_h = resize_pixel[0]
        save_frame_w = resize_pixel[1]

    os.makedirs(save_path, exist_ok=True)
    
    # if not isinstance(resize_ratio, int):
    #     raise TypeError(f"data path: {data_path} is not a string")
    
    """
    Replace \v, \t with /v, /t 
    Convert path str to byte?
    """
    # data_path_copy = data_path.encode('unicode_escape').decode('utf-8')
    # if "\\" in data_path_copy:
    #     data_path = 
    
    if Config.face_detector == "mediapipe":
        crop_worker = MPcrop(Config.default_crop_size, Config.resize_ratio)
    
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
    frame_start_num = 10**len(str(max_frame))

    with tqdm(total=total_frames, desc="Preprocessing all videos with 1 process", unit='frame') as pbar:
        for i, video in enumerate(video_list):

            frame_count = frame_start_num
            cap = cv2.VideoCapture(os.path.join(data_path, video))
            
            frame_shape = [cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
            
            if not cap.isOpened():
                raise ValueError(f"\n*** Failed to open the video: {video} ***")
            
            print("Processing video: {video} ({frame_num_list[i]} frames)")
            
            last_bbox = []
            save_chunk = np.zeros((frame_chunk, save_frame_h, save_frame_w, 3))
            frame_chunk_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count == frame_start_num:
                    crop_frame, last_bbox = crop_worker.face_crop(frame, frame_shape, frame_count)
                else:
                    crop_frame, last_bbox = crop_worker.face_crop(frame, frame_shape, frame_count, last_bbox)
                
                if resize_ratio > 1.0:
                    crop_frame = cv2.resize(crop_frame, (0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_CUBIC)
                elif resize_ratio < 1.0:
                    crop_frame = cv2.resize(crop_frame, (0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)
                
                if resize_pixel[0] != 1 or resize_pixel[1] != 1:
                    crop_frame = cv2.resize(crop_frame, resize_pixel, fx=0, fy=0, interpolation=cv2.INTER_AREA)

                """ Save as npy files with chunk size in Preprocessing_Config"""

                if frame_count%frame_chunk == 0:
                    if frame_count == frame_start_num:
                        save_chunk[frame_chunk_index] = crop_frame        
                    np.save(os.path.join(save_path, f"{frame_count}.npy"), save_chunk)
                    frame_chunk_index = 0
                save_chunk[frame_chunk_index] = crop_frame
                
                # save_path = os.path.join(save_path, f"{frame_count}.png")
                # cv2.imwrite(save_path, crop_frame)
                
                pbar.update(1)
                frame_count += 1
            cap.release()
        
        print("\nFinished preprocessing all videos.")
    
    crop_worker.close()


def video_multi_preprocess(Config):
    """
    Parameters
    ----------
    Config : dataclass
        Refer the 'config.py' file.

    Returns
    -------
    Nothing

    Purpose
    -------
    Crop a face from a video or image and save it as npy files with multiprocess.
    
    """
    video_list = os.listdir(Config.data_path)
    
    max_frame = 0
    for video in video_list:
        cap = cv2.VideoCapture(os.path.join(Config.data_path, video))
        if not cap.isOpened():
            print(f"\n*** Failed to open the video: {video} ***")
            continue
        number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frame = max(max_frame, number_of_frames)
    
    frame_start_num = 10 ** len(str(max_frame))


def resize_and_pad(image, default_size):
    h, w = image.shape[:2]
    target_h, target_w = default_size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    padded_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    padded_image[pad_y:pad_y+new_h, pad_x:pad_x+new_w, :] = resized_image
    return padded_image


class MPcrop:
    def __init__(self, default_size, resize_ratio):
        self.face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.last_bbox = None
        self.default_size = default_size
        self.resize_ratio = resize_ratio
    
    def close(self):
        self.face_detector.close()
    
    def face_crop(self, image, frame_shape, frame_index, last_bbox=None):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        iw, ih = frame_shape
        results = self.face_detector.process(image_rgb)
    
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x_min = int(bboxC.xmin * iw)
                y_min = int(bboxC.ymin * ih)
                width = int(bboxC.width * iw)
                height = int(bboxC.height * ih)
                last_bbox = (x_min, y_min, width, height)
                break
        elif last_bbox:
            print(f"Failed to detect the face at frame: ({frame_index}), using last bbox")
            x_min, y_min, width, height = last_bbox
        else:
            raise ValueError("Failed to detect the face at the first frame")
    
        cropped_face = image_rgb[y_min:y_min + height, x_min:x_min + width]
        
        default_cropped_face = resize_and_pad(cropped_face, self.default_size)
        resized_cropped_face = cv2.resize(default_cropped_face, (0, 0), fx=self.resize_ratio, fy=self.resize_ratio, interpolation=cv2.INTER_LINEAR)
    
        return resized_cropped_face, last_bbox


def crop_check(Config):
    import matplotlib.pyplot as plt
    import time
    frame_list = os.listdir(os.path.join(Config.cache_path, 'video'))
    frame_chunk = int(Config.data_chunk*Config.minimum_chunk["video"])
    for i, frame in enumerate(frame_list):
        temp_load_npy = np.load(os.path.join(Config.cache_path, f"video/{frame}")).astype(np.uint8)

        for j in range(frame_chunk):
            plt.imshow(temp_load_npy[j])
            plt.title(f"frame_index: {frame_chunk*i+j}")
            plt.savefig(os.path.join(Config.cache_path, f"video/{frame.split('.')[0]}.png"))
            plt.close()


if __name__ == "__main__":
    
    from dataclasses import dataclass, field
    
    @dataclass
    class direct_use_config:
        """ Fixed by the sampling rates of each sensors """
        minimum_chunk: dict = field(default_factory=lambda: {"eeg_label":120, "ecg":26, "ppg":27, "video":6})
        
        preprocess: bool = True
        data_path: str = "D:/home/BCML/IITP-multimodal/data/test_data"  # this path should be changed
        cache_path: str = "D:/home/BCML/IITP-multimodal/data/preprocessed_data"  # this path should be changed
        input_features: list[str] = field(default_factory=list)
        multi_process: int = 1  # multiprocessding process number
        face_detector: str = "mediapipe"
        default_crop_size: list[int, int] = field(default_factory=list)
        resize_ratio: float = 1.0
        resize_pixel: tuple[int, int] = field(default_factory=lambda: (1, 1))
        data_chunk: float = 0.2
        match_files: list[str] = field(default_factory=list)
        extensions: dict = field(default_factory=lambda: {'eeg': '.mat', 'ecg': '.csv', 'rppg': '.npy', 'video': '.mp4'})
        dataset: str = "16channel_kw"

        @classmethod
        def only_video(cls):
            return cls(input_features=['video'],
                    default_crop_size=[640, 640],
                    resize_pixel=(48,48),)
        
        @classmethod
        def only_video_multi(cls):
            return cls(input_features=['video'],
                    default_crop_size=[640, 640],
                    resize_pixel=(48,48),
                    multi_process=4,)

    direct_use_config = direct_use_config.only_video_multi()
    video_preprocess(direct_use_config)
        
    # crop_check(direct_use_config)