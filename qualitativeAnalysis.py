import numpy as np
import cv2
import time
from skimage import io 
from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt
from numpy import expand_dims
import argparse
import os
import models
from dataGenerator import *
from datasetProcess import *

### HOW TO RUN
# python featureMapVisualization.py --weights WEIGHTS_PATH --video INPUT_VIDEO_PATH

def background_suppression(data):
    video = np.array(data, dtype = np.float32)
    avgBack = np.mean(video, axis=0)
    video = np.abs(video - avgBack)
    return video

def frame_difference(video):
    num_frames = len(video)
    k = 1
    out = []
    for i in range(num_frames - k):
        out.append(video[i+k] - video[i])
    return np.array(out,dtype=np.float32)

def normalize(data):
    data = (data / 255.0).astype(np.float32)
    mean = np.mean(data)
    std = np.std(data)
    return (data-mean) / std

def crop_center(video, x_crop=10, y_crop=30):
    frame_size = np.size(video, axis=1)
    x = frame_size
    y = frame_size
    x_start = x_crop
    x_end = x - x_crop
    y_start = y_crop
    y_end = y-y_crop
    video = video[:, y_start:y_end, x_start:x_end, :]
    return video

def saveVideo(file, name, dest, asFrames = False, fps = 29):
    if file.dtype != np.uint8:    
        file = np.array(file, dtype = np.uint8)
    outpath = os.path.join(dest, name)
    _, h, w, _ = file.shape
    size = (h, w)
    if asFrames:
        os.mkdir(outpath)
        print("saving frames to ", outpath)
        print("number of frames:",len(file))
        for i in range(len(file)):
            filename = os.path.join(outpath, str(i)+".png")
            frame = cv2.cvtColor(file[i], cv2.COLOR_BGR2RGB)
            cv2.imwrite(filename, frame)
    else:
        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        print("saving video to ", outpath)
        out = cv2.VideoWriter(outpath,fourcc, fps, size)
        print("video length:",len(file))
        for i in range(len(file)):
            out.write(file[i])
        out.release()


def qualitative(args):
    mode = "both"
    dataset = 'rwf2000'
    vid_len = 32
    dataset_frame_size = 320
    input_frame_size = 224
    frame_diff_interval = 1
    one_hot = False
    lstm_type = 'sepconv' 
    preprocess_data = False
    if preprocess_data:
        if dataset == 'rwf2000':
            os.mkdir(os.path.join(dataset, 'processed'))
            convert_dataset_to_npy(src='{}/RWF-2000'.format(dataset), dest='{}/processed'.format(
                dataset), crop_x_y=None, target_frames=vid_len, frame_size= dataset_frame_size)

    test_generator = DataGenerator(directory='{}/processed/test'.format(dataset),
                                batch_size=1,
                                data_augmentation=False,
                                shuffle=True,
                                one_hot=one_hot,
                                sample=False,
                                resize=input_frame_size,
                                target_frames = vid_len,
                                frame_diff_interval = frame_diff_interval,
                                dataset = dataset,
                                normalize_ = False,
                                background_suppress = False,
                                mode = mode)


    print('> getting the model from...', args["weights"])  
    model =  models.getProposedModelM(size=224, seq_len=32, frame_diff_interval = 1, mode="both", lstm_type=lstm_type)
    model.load_weights(args["weights"]).expect_partial()
    model.trainable = False
    model.summary()
    evaluate(model, test_generator, args["outputPath"])


def evaluate(model, datagen, dest, count = 100):
    classes = {0:"violent", 1:"nonviolent"}
    for i, (x,y) in enumerate(datagen):
        if i == count:
            break
        data = x[0]; target = y[0]
        if i == 0:
            print(data.shape)
        data = np.squeeze(data)
        p = model.predict(x)
        p = np.squeeze(p)
        if p >= 0.50:
            predicted = 1
        else:
            predicted = 0   
        print("> index:",i, " target:",target, " predicted:",predicted)
        saveVideo(data, str(i)+"_GT:"+str(classes[target])+"_PL:"+str(classes[predicted]), dest, asFrames = True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w","--weights", required=True, help="path to the weights")
    ap.add_argument("-o","--outputPath", default="/content/qualitativeAnalysis", help="path for saving output")
    args = vars(ap.parse_args())
    qualitative(args)

main()