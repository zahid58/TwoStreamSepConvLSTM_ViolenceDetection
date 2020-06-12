import os
import math
import random
from sklearn.model_selection import KFold
import shutil
import cv2
from tqdm import tqdm
import numpy as np

def train_test_split(dataset_name=None,source=None,test_ratio=.20):
    assert (dataset_name=='hockey' or dataset_name=='movies')
    fightVideos = []
    nonFightVideos = []
    for filename in os.listdir(source):
        filepath = os.path.join(source, filename)
        if filename.endswith('.avi') or filename.endswith('.mpg'):
            if dataset_name=='hockey':
                if filename.startswith('fi'):
                    fightVideos.append(filepath)
                else:
                    nonFightVideos.append(filepath)
            if dataset_name=='movies':
                if 'fi' in filename:
                    fightVideos.append(filepath)
                else:
                    nonFightVideos.append(filepath)
    random.seed(42)
    random.shuffle(fightVideos)
    random.shuffle(nonFightVideos)
    fight_len = len(fightVideos)
    split_index = fight_len - (fight_len*test_ratio)
    trainFightVideos = fightVideos[:split_index]
    testFightVideos = fightVideos[split_index:]
    trainNonFightVideos = nonFightVideos[:split_index]
    testNonFightVideos = nonFightVideos[split_index:]
    split = trainFightVideos, testFightVideos, trainNonFightVideos, testNonFightVideos
    return split


def five_fold_split(dataset_name,source):
    assert (dataset_name=='hockey' or dataset_name=='movies')
    fightVideos = []
    nonFightVideos = []
    for filename in os.listdir(source):
        filepath = os.path.join(source, filename)
        if filename.endswith('.avi') or filename.endswith('.mpg'):
            if dataset_name=='hockey':
                if filename.startswith('fi'):
                    fightVideos.append(filepath)
                else:
                    nonFightVideos.append(filepath)
            if dataset_name=='movies':
                if 'fi' in filename:
                    fightVideos.append(filepath)
                else:
                    nonFightVideos.append(filepath)
    random.seed(42)
    random.shuffle(fightVideos)
    random.shuffle(nonFightVideos)
    kf = KFold(n_splits=5,random_state=42,shuffle=True)
    splits = []
    for train_ind, test_ind in kf.split(fightVideos):
        testFightVideos = [fightVideos[i] for i in test_ind]
        trainFightVideos = [fightVideos[i] for i in train_ind]
        trainNonFightVideos = [nonFightVideos[i] for i in train_ind]
        testNonFightVideos = [nonFightVideos[i] for i in test_ind]
        splits.append((trainFightVideos,testFightVideos,trainNonFightVideos,testNonFightVideos))
    return splits


def move_train_test(dest, data):
    trainFightVideos, testFightVideos, trainNonFightVideos, testNonFightVideos = data
    trainPath = os.path.join(dest,'train')
    testPath = os.path.join(dest,'test')
    os.mkdir(trainPath)
    os.mkdir(testPath)
    trainFightPath = os.path.join(trainPath,'fight')
    trainNonFightPath = os.path.join(trainPath,'nonFight')
    testFightPath = os.path.join(testPath,'fight')
    testNonFightPath = os.path.join(testPath,'nonFight')
    os.mkdir(trainFightPath)
    os.mkdir(trainNonFightPath)
    os.mkdir(testFightPath)
    os.mkdir(testNonFightPath)
    print("moving files...")
    for filepath in trainFightVideos:
        shutil.copy(filepath,trainFightPath)
    print(len(trainFightVideos),'files have been copied to',trainFightPath)
    for filepath in testFightVideos:
        shutil.copy(filepath,testFightPath)
    print(len(trainNonFightVideos),'files have been copied to',trainNonFightPath)
    for filepath in trainNonFightVideos:
        shutil.copy(filepath,trainNonFightPath)
    print(len(testFightVideos),'files have been copied to',testFightPath)
    for filepath in testNonFightVideos:
        shutil.copy(filepath,testNonFightPath)
    print(len(testNonFightVideos),'files have been copied to',testNonFightPath)

def crop_img_remove_black(img,x_crop,y_crop,y,x):
    x_start = x_crop
    x_end = x - x_crop
    y_start = y_crop
    y_end = y-y_crop
    frame = img[y_start:y_end,x_start:x_end,:]
    #return img[44:244,16:344, :]
    return frame


def Video2Npy(file_path, resize=(224,224),crop_x_y=None):
    """Load video and tansfer it into .npy format
    Args:
        file_path: the path of video file
        resize: the target resolution of output video
    Returns:
        frames: gray-scale video
        flows: magnitude video of optical flows 
    """
    # Load video
    cap = cv2.VideoCapture(file_path)
    # Get number of frames
    len_frames = int(cap.get(7))
    # Extract frames from video
    try:
        frames = []
        for i in range(len_frames-1):
            _, x_ = cap.read()
            if crop_x_y:
                frame = crop_img_remove_black(x_,crop_x_y[0],crop_x_y[1],x_.shape[0],x_.shape[1])
            else:
                frame = x_
            frame = cv2.resize(frame,resize, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.reshape(frame, (224,224,3))
            frames.append(frame)   
    except:
        print("Error: ", file_path, len_frames,i)
    finally:
        frames = np.array(frames)
        cap.release()        
    return frames


def Save2Npy(file_dir, save_dir,crop_x_y=None):
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
        data = Video2Npy(file_path=video_path, resize=(224,224),crop_x_y=crop_x_y)
        data = np.uint8(data)
        # Save as .npy file
        np.save(save_path, data)
    return None


def convert_dataset_to_npy(src,dest,crop_x_y=None):
    if not os.path.isdir(dest):
        os.path.mkdir(dest)
    for dir_ in ['train','test']:
        for cat_ in ['fight','nonFight']:
          path1 = os.path.join(src, dir_, cat_)
          path2 = os.path.join(dest, dir_, cat_ )
          Save2Npy(file_dir=path1, save_dir=path2,crop_x_y=crop_x_y)


