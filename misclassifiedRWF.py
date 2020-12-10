import os
os.environ['PYTHONHASHSEED'] = '42'
import numpy as np
from numpy.random import seed, shuffle
from random import seed as rseed
from tensorflow.random import set_seed
import cnn_lstm_models
seed(42)
rseed(42)
set_seed(42)
import random
from customLayers import SepConvLSTM2D
import pickle
import shutil
from utils import *
from dataGenerator import *
from datasetProcess import *
from tensorflow.keras.models import load_model
import pandas as pd
import os
import cv2

#-----------------------------------
mode = "both"
dataset = 'rwf2000'
crop_dark = {
    'rwf2000': (0, 0),
}
vid_len = 32
dataset_frame_size = 320
input_frame_size = 224
frame_diff_interval = 1
one_hot = False
loss = 'binary_crossentropy'
lstm_type = 'attensepconv' # conv

###################################################

preprocess_data = False

bestValPath =  '/gdrive/My Drive/THESIS/Data/attenSepConvLSTMmodel/' + \
    str(dataset) + '_best_val_acc_Model'   
 
###################################################

if preprocess_data:

    if dataset == 'rwf2000':
        os.mkdir(os.path.join(dataset, 'processed'))
        convert_dataset_to_npy(src='{}/RWF-2000'.format(dataset), dest='{}/processed'.format(
            dataset), crop_x_y=None, target_frames=vid_len, frame_size= dataset_frame_size)


# train_generator = DataGenerator(directory='{}/processed/train'.format(dataset),
#                                 batch_size=1,
#                                 data_augmentation=False,
#                                 shuffle=False,
#                                 one_hot=one_hot,
#                                 sample=False,
#                                 resize=input_frame_size,
#                                 target_frames = vid_len,
#                                 frame_diff_interval = frame_diff_interval,
#                                 dataset = dataset,
#                                 normalize_ = False,
#                                 mode = mode)

test_generator = DataGenerator(directory='{}/processed/test'.format(dataset),
                               batch_size=1,
                               data_augmentation=False,
                               shuffle=False,
                               one_hot=one_hot,
                               sample=False,
                               resize=input_frame_size,
                               target_frames = vid_len,
                               frame_diff_interval = frame_diff_interval,
                               dataset = dataset,
                               normalize_ = False,
                               mode = mode)

#--------------------------------------------------

print('> getting the model from...', bestValPath)
model =  cnn_lstm_models.getProposedModel(size=input_frame_size, seq_len=vid_len, frame_diff_interval = frame_diff_interval, mode="both", lstm_type=lstm_type)
model.load_weights(bestValPath)

#--------------------------------------------------

SavePath = '/gdrive/My Drive/THESIS/Data/results/' + str(dataset)+'/'

#--------------------------------------------------

def avg_back_frame_diff(data):
    if data.dtype != np.float32:
        data = np.array(data, dtype = np.float32)
    num_frames = len(data)
    avgBack = np.sum(data, axis=0)
    avgBack /= num_frames
    kernel = np.ones((3,3),np.uint8)
    for i in range(num_frames):
        frame = abs(data[i] - avgBack)
        frame = np.array(frame, dtype = np.uint8)
        frame = cv2.GaussianBlur(frame,(3,3),0)
        g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        _, g = cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        g = cv2.morphologyEx(g , cv2.MORPH_OPEN, kernel)
        frame = cv2.bitwise_and(frame,frame, mask= g)	    
        data[i] = frame
    return np.array(data,dtype=np.float32)    

def normalize(data):
    data = (data / 255.0).astype(np.float32)
    mean = np.mean(data)
    std = np.std(data)
    return (data-mean) / std

def saveVideo(file, name, dest, fps = 29):
    if file.dtype != np.uint8:    
        file = np.array(file, dtype = np.uint8)
    outpath = os.path.join(dest, name)
    _, h, w, _ = file.shape
    size = (h, w)
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    print("saving video to ", outpath)
    out = cv2.VideoWriter(outpath,fourcc, fps, size)
    for i in range(len(file)):
       out.write(file[i])
    out.release()

def evaluate(model, datagen, dest):
    num_mis = 0
    total = len(datagen)
    for i, (x,y) in enumerate(datagen):
        data = x[0]; target = y[0] ; diff_data = x[1]
        x_ = [normalize(data), normalize(diff_data)]
        data = np.squeeze(data); diff_data = np.squeeze(diff_data)
        p = model.predict(x_)
        p = np.squeeze(p)
        if p >= 0.50:
            predicted = 1
        else:
            predicted = 0
        if i == 0:
            print("data shape:", data.shape)
            print("target:",target, " predicted:",predicted)
        if( predicted != target ):
            print('----------------------------------')
            num_mis += 1
            print("misclassification at file ", str(i))
            print("target:",target, " predicted:",predicted)
            if target == 1:
                label = "NonViolent"
            else:
                label = "Violent"
            saveVideo(data, "frames_"+str(i)+"_"+label+".avi", dest)
            diff_data = new_frame_diff(data)
            saveVideo(diff_data, "frameDifference_"+str(i)+"_"+label+".avi", dest)
    print("total: ", total, " wrong: ",num_mis, " right: ", total - num_mis, " accuracy: ", np.round((total-num_mis)/total,5))

path = "/content/violenceDetection/misClassified"
train_path = os.path.join(path, "train")
test_path = os.path.join(path, "test")
if os.path.isdir(path):
    shutil.rmtree(path)
os.mkdir(path)
os.mkdir(train_path)
os.mkdir(test_path)

# print("on train...")
# evaluate(model, train_generator, train_path)

print("on test...")
evaluate(model, test_generator, test_path)

#----------------------------------------------------------




