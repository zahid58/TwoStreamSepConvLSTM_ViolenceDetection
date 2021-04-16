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

def uniform_sampling(video, target_frames=20):
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
    return np.array(sampled_video, dtype=np.float32)

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

def saveVideo(file, name, dest, fps = 29):
    if file.dtype != np.uint8:    
        file = np.array(file, dtype = np.uint8)
    outpath = os.path.join(dest, name)
    _, h, w, _ = file.shape
    size = (h, w)
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    print("saving video to ", outpath)
    out = cv2.VideoWriter(outpath,fourcc, fps, size)
    print("video length:",len(file))
    for i in range(len(file)):
       out.write(file[i])
    out.release()


def visualize(args):
    # load the processed .npy files
    video = np.load(args["video"], mmap_mode='r')
    video = np.float32(video)
    video = crop_center(video, x_crop=(320-224)//2, y_crop=(320-224)//2)
    output_video_path = os.path.basename(args["video"])
    saveVideo(video, output_video_path+".avi", args["outputPath"])
    diff_data = frame_difference(video)
    data = background_suppression(video)
    diff_data = normalize(diff_data);  data = normalize(data)
    diff_data = np.expand_dims(diff_data, axis=0); data = np.expand_dims(data, axis=0)
    x_ = [data, diff_data]


    print('> getting the model from...', args["weights"])  
    model =  models.getProposedModelM(size=224, seq_len=32, frame_diff_interval = 1, mode="both", lstm_type="sepconv")
    model.load_weights(args["weights"]).expect_partial()
    model.trainable = False
    model.summary()
    indexes = [5]  
    # index of the layers at the end of each conv block  (Conv>Activation>BN) of trafficSignNetModel
    num_features = [8]
    outputs = [model.layers[i].output for i in indexes]
    model2 = Model(inputs=model.inputs, outputs=outputs)
    feature_maps = model2.predict(x_)

    for ind,fmap in enumerate(feature_maps):
        ix = 1
        figtitle = 'output_of_layer_'+ str(indexes[ind],) + '_' + model.layers[indexes[ind]].name  
        figtitle = os.path.join(args["outputPath"], figtitle)
        print(figtitle)
        for _ in range(4):
            for _ in range(num_features[ind]):
                ax = plt.subplot(4,num_features[ind],ix)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(fmap[0,:,:,ix-1], cmap = 'gray')
                ix += 1
        plt.savefig(figtitle + ".jpg")
        plt.show()
    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w","--weights", required=True, help="path to the weights")
    ap.add_argument("-v","--video", required=True, help="path to the video")
    ap.add_argument("-o","--outputPath", default="/content/featMapVis", help="path for saving output")
    args = vars(ap.parse_args())
    visualize(args)

main()