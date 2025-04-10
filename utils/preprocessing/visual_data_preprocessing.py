import os
import numpy as np
import cv2
from tqdm import tqdm
import mediapipe as mp
from pymediainfo import MediaInfo

def get_video_creation_time(Config):
    for video in Config.match_files:
        media_info = MediaInfo.parse(os.path.join(Config.data_path, f"video/{video}{Config.extensions['video']}"))
        for track in media_info.tracks:
            if track.track_type == "General":
                return track.tagged_date
    return None

def image_preprocess(Config):
    return 0

def image_multi_preprocess(Config):
    return 0

def video_preprocess(Config):
    
    data_path = os.path.join(Config.data_path, 'video')
    save_path = os.path.join(Config.cache_path, "video")
    resize_ratio = Config.resize_ratio
    crop_size = Config.default_crop_size
    
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
    frame_start_num = (max_frame // 10 + 1) * 10
    
    with tqdm(total=total_frames, desc="Preprocessing all videos with 1 process", unit='frame') as pbar:
        for i, video in enumerate(video_list):
            frame_count = frame_start_num
            cap = cv2.VideoCapture(os.path.join(data_path, video))
            
            frame_shape = [cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
            
            if not cap.isOpened():
                raise ValueError(f"\n*** Failed to open the video: {video} ***")
            
            print("Processing video: {video} ({frame_num_list[i]} frames)")
            
            last_bbox = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count == frame_start_num:
                    crop_frame, last_bbox = crop_worker.face_crop(frame, frame_shape, frame_count)
                else:
                    crop_frame, last_bbox = crop_worker.face_crop(frame, frame_shape, frame_count, last_bbox)
                
                if resize_ratio != 1.0:
                    crop_frame = cv2.resize(crop_frame, (0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
                save_path = os.path.join(save_path, f"{frame_count}.png")
                
                """ Save as npy files with chunk size in Preprocessing_Config"""
                cv2.imwrite(save_path, crop_frame)
                
                pbar.update(1)
                frame_count += 1
            cap.release()
        
        print("\nFinished preprocessing all videos.")
    
    crop_worker.close()
    
def video_multi_preprocess(data_path, save_path, resize_ratio, crop_size):
    video_list = os.listdir(data_path)
    
    max_frame = 0
    for video in video_list:
        cap = cv2.VideoCapture(os.path.join(data_path, video))
        if not cap.isOpened():
            print(f"\n*** Failed to open the video: {video} ***")
            continue
        number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frame = max(max_frame, number_of_frames)
    
    frame_start_num = (max_frame // 10 + 1) * 10

"""
# It has better readability to find minimum scale ratio...
# GPT won.

def resize_default_size(image, default_size):
    ih, iw, _ = image.shape
    max_axis_pixel = max(ih, iw)
    max_axis = image.shape.index(max_axis_pixel)
        
    def padding():
        
    
    if max_axis == 0:  # if the ih is larger than the iw
        new_w = int(another_size*default_size/ih)
        new_h = default_size
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        new_w = default_size
        new_h = int(another_size*default_size/iw)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
"""

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

