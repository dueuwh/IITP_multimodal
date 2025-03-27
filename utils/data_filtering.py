"""preprocessing code made by junghwan"""

import os
import csv
import cv2
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
import mediapipe as mp

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

def process_ecg_rppg(subject_ids, base_ecg_path, base_video_path, output_ecg_csv, output_rppg_npy):
    def get_files(directory, ext):
        return sorted([f for f in os.listdir(directory) if f.endswith(ext)])

    def flatten_list(nested_list):
        return [item for sublist in nested_list for item in sublist]

    def bandpass_filter(data, lowcut, highcut, fs, method='mne', order=2):
        nyquist = 0.5 * fs
        low, high = lowcut / nyquist, highcut / nyquist

        if method == 'mne' and MNE_AVAILABLE:
            return mne.filter.filter_data(data, sfreq=fs, l_freq=lowcut, h_freq=highcut, method='fir')
        elif method == 'fft':
            freqs = np.fft.fftfreq(len(data), d=1/fs)
            fft_data = np.fft.fft(data)
            mask = (lowcut <= np.abs(freqs)) & (np.abs(freqs) <= highcut)
            fft_data[~mask] = 0
            return np.fft.ifft(fft_data).real
        else:
            b, a = signal.butter(order, [low, high], btype='band')
            return signal.filtfilt(b, a, data)

    def extract_rppg_from_video(video_path):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        rppg_signal = []

        with mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)

                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                        forehead_region = frame[y:y + h_box // 3, x:x + w_box]
                        if forehead_region.size == 0:
                            continue
                        avg_color = np.mean(forehead_region, axis=(0, 1))
                        rppg_signal.append(avg_color[1])
        cap.release()

        if len(rppg_signal) > fps * 3:
            return bandpass_filter(np.array(rppg_signal), 0.5, 4.0, fps), fps
        else:
            return None, None

    ECG_LIST, VIDEO_LIST = [], []
    for subj in subject_ids:
        ecg_dir = os.path.join(base_ecg_path, str(subj))
        video_dir = os.path.join(base_video_path, str(subj))
        ECG_LIST.append([os.path.join(ecg_dir, f) for f in get_files(ecg_dir, '.csv')])
        VIDEO_LIST.append([os.path.join(video_dir, f) for f in get_files(video_dir, '.mp4')])

    ECG_LIST, VIDEO_LIST = flatten_list(ECG_LIST), flatten_list(VIDEO_LIST)
    TIME_LIST, ECG_RAW_LIST = [], []

    for ecg_file in ECG_LIST:
        time, ecg = [], []
        with open(ecg_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                time.append(row[0])
                ecg.append(float(row[1]))
        TIME_LIST.append(time)
        ECG_RAW_LIST.append(ecg)

    fs, lowcut, highcut = 120, 1.0, 30.0
    filtered_ECG_LIST = [bandpass_filter(ecg, lowcut, highcut, fs, method='mne') for ecg in ECG_RAW_LIST]

    ecg_data = []
    for i, ecg_signal in enumerate(filtered_ECG_LIST):
        for j, value in enumerate(ecg_signal):
            ecg_data.append({"ECG_Index": subject_ids[i], "Sample": j, "Amplitude": value})
    pd.DataFrame(ecg_data).to_csv(output_ecg_csv, index=False)
    print(f"Filtered ECG signals saved to {output_ecg_csv}")

    rPPG_RESULTS = []
    for video_file in VIDEO_LIST:
        rppg_signal, fps = extract_rppg_from_video(video_file)
        if rppg_signal is not None:
            rPPG_RESULTS.append({"file": video_file, "rppg_signal": rppg_signal, "fps": fps})

    for result in rPPG_RESULTS:
        npy_filename = os.path.basename(result["file"]).replace(".mp4", "_rppg.npy")
        np.save(npy_filename, result["rppg_signal"])
        print(f"rPPG signal saved to {npy_filename}")

    return output_ecg_csv, [res["file"] for res in rPPG_RESULTS]

if __name__ == "__main__":
    # 실행 예제
    subject_ids = [160 + i for i in range(3)]
    base_ecg_path = "/content/drive/MyDrive/testCode/16channel_Emotion/Polar/"
    base_video_path = "/content/drive/MyDrive/testCode/16channel_Emotion/Video/"
    output_ecg_csv = "filtered_ecg_signals.csv"
    output_rppg_npy = "rppg_signals"
    
    process_ecg_rppg(subject_ids, base_ecg_path, base_video_path, output_ecg_csv, output_rppg_npy)
