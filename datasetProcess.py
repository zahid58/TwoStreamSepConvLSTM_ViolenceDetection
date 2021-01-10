import os
import math
import random
from sklearn.model_selection import KFold
import shutil
import cv2
from tqdm import tqdm
import numpy as np


def train_test_split(dataset_name=None, source=None, test_ratio=.20):
    assert (dataset_name == 'hockey' or dataset_name == 'movies' or dataset_name == 'surv')
    fightVideos = []
    nonFightVideos = [] 
    for filename in os.listdir(source):
        filepath = os.path.join(source, filename)
        if filename.endswith('.avi') or filename.endswith('.mpg') or filename.endswith('.mp4'):
            if dataset_name == 'hockey':
                if filename.startswith('fi'):
                    fightVideos.append(filepath)
                else:
                    nonFightVideos.append(filepath)
            elif dataset_name == 'movies':
                if 'fi' in filename:
                    fightVideos.append(filepath)
                else:
                    nonFightVideos.append(filepath)
    random.seed(0)
    random.shuffle(fightVideos)
    random.shuffle(nonFightVideos)
    fight_len = len(fightVideos)
    split_index = int(fight_len - (fight_len*test_ratio))
    trainFightVideos = fightVideos[:split_index]
    testFightVideos = fightVideos[split_index:]
    trainNonFightVideos = nonFightVideos[:split_index]
    testNonFightVideos = nonFightVideos[split_index:]
    split = trainFightVideos, testFightVideos, trainNonFightVideos, testNonFightVideos
    return split


def move_train_test(dest, data):
    trainFightVideos, testFightVideos, trainNonFightVideos, testNonFightVideos = data
    trainPath = os.path.join(dest, 'train')
    testPath = os.path.join(dest, 'test')
    os.mkdir(trainPath)
    os.mkdir(testPath)
    trainFightPath = os.path.join(trainPath, 'fight')
    trainNonFightPath = os.path.join(trainPath, 'nonFight')
    testFightPath = os.path.join(testPath, 'fight')
    testNonFightPath = os.path.join(testPath, 'nonFight')
    os.mkdir(trainFightPath)
    os.mkdir(trainNonFightPath)
    os.mkdir(testFightPath)
    os.mkdir(testNonFightPath)
    print("moving files...")
    for filepath in trainFightVideos:
        shutil.copy(filepath, trainFightPath)
    print(len(trainFightVideos), 'files have been copied to', trainFightPath)
    for filepath in testFightVideos:
        shutil.copy(filepath, testFightPath)
    print(len(trainNonFightVideos), 'files have been copied to', trainNonFightPath)
    for filepath in trainNonFightVideos:
        shutil.copy(filepath, trainNonFightPath)
    print(len(testFightVideos), 'files have been copied to', testFightPath)
    for filepath in testNonFightVideos:
        shutil.copy(filepath, testNonFightPath)
    print(len(testNonFightVideos), 'files have been copied to', testNonFightPath)


def crop_img_remove_black(img, x_crop, y_crop, y, x):
    x_start = x_crop
    x_end = x - x_crop
    y_start = y_crop
    y_end = y-y_crop
    frame = img[y_start:y_end, x_start:x_end, :]
    # return img[44:244,16:344, :]
    return frame


def uniform_sampling(video, target_frames=64):
    # get total frames of input video and calculate sampling interval
    len_frames = video.shape[0]
    interval = int(np.ceil(len_frames/target_frames))
    # init empty list for sampled video and
    sampled_video = []
    for i in range(0, len_frames, interval):
        sampled_video.append(video[i])
    # calculate numer of padded frames and fix it
    num_pad = target_frames - len(sampled_video)
    padding = []
    if num_pad > 0:
        for i in range(-num_pad, 0):
            try:
                padding.append(video[i])
            except:
                padding.append(video[0])
        sampled_video += padding
    # get sampled video
    return np.array(sampled_video)


def Video2Npy(file_path, resize=320, crop_x_y=None, target_frames=None):
    """Load video and tansfer it into .npy format
    Args:
        file_path: the path of video file
        resize: the target resolution of output video
        crop_x_y: black boundary cropping
        target_frames:
    Returns:
        frames: gray-scale video
        flows: magnitude video of optical flows 
    """
    # Load video
    cap = cv2.VideoCapture(file_path)
    # Get number of frames
    len_frames = int(cap.get(7))
    frames = []
    try:
        for i in range(len_frames):
            _, x_ = cap.read()
            if crop_x_y:
                frame = crop_img_remove_black(
                    x_, crop_x_y[0], crop_x_y[1], x_.shape[0], x_.shape[1])
            else:
                frame = x_
            frame = cv2.resize(frame, (resize,resize), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.reshape(frame, (resize, resize, 3))
            frames.append(frame)
    except Exception as e:
        print("Error: ", file_path, len_frames)
        print(e)
    finally:
        frames = np.array(frames)
        cap.release()
    frames = uniform_sampling(frames, target_frames=target_frames)
    return frames


def Save2Npy(file_dir, save_dir, crop_x_y=None, target_frames=None, frame_size=320):
    """Transfer all the videos and save them into specified directory
    Args:
        file_dir: source folder of target videos
        save_dir: destination folder of output .npy files
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # List the files
    videos = os.listdir(file_dir)
    for v in tqdm(videos):
        # Split video name
        video_name = v.split('.')[0]
        # Get src
        video_path = os.path.join(file_dir, v)
        # Get dest
        save_path = os.path.join(save_dir, video_name+'.npy')
        # Load and preprocess video
        data = Video2Npy(file_path=video_path, resize=frame_size,
                         crop_x_y=crop_x_y, target_frames=target_frames)
        if target_frames:
            assert (data.shape == (target_frames,
                                   frame_size, frame_size, 3))
        os.remove(video_path)
        data = np.uint8(data)
        # Save as .npy file
        np.save(save_path, data)
    return None


def convert_dataset_to_npy(src, dest, crop_x_y=None, target_frames=None, frame_size=320):
    if not os.path.isdir(dest):
        os.path.mkdir(dest)
    for dir_ in ['train', 'test']:
        for cat_ in ['fight', 'nonFight']:
            path1 = os.path.join(src, dir_, cat_)
            path2 = os.path.join(dest, dir_, cat_)
            Save2Npy(file_dir=path1, save_dir=path2, crop_x_y=crop_x_y,
                     target_frames=target_frames, frame_size=frame_size)