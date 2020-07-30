import random
from customLayers import SepConvLSTM2D
import pickle
import shutil
import sepConvLstmNet
import rwfRGBonly
from utils import *
from dataGenerator import *
from datasetProcess import *
from tensorflow.keras.models import load_model
from tensorflow.random import set_seed
import os
import pandas as pd
from numpy.random import seed, shuffle
seed(42)
random.seed(42)
set_seed(42)

#-----------------------------------

dataset = 'rwf2000'
crop_dark = {
    'rwf2000': (0, 0),
}
batch_size = 4
vid_len = 32
dataset_frame_size = 320
input_frame_size = 224
frame_diff_interval = 1
###################################################

preprocess_data = False

bestModelPath = '/gdrive/My Drive/THESIS/Data/' + \
    str(dataset) + '_bestModel.h5'

bestValPath =  '/gdrive/My Drive/THESIS/Data/' + \
    str(dataset) + '_best_val_acc_Model.h5'   
 
###################################################

if preprocess_data:
    os.mkdir(os.path.join(dataset, 'processed'))
    convert_dataset_to_npy(src='{}/RWF-2000'.format(dataset), dest='{}/processed'.format(
        dataset), crop_x_y=None, target_frames=vid_len, frame_size= dataset_frame_size)


train_generator = DataGenerator(directory='{}/processed/train'.format(dataset),
                                batch_size=batch_size,
                                data_augmentation=False,
                                shuffle=False,
                                one_hot=False,
                                sample=False,
                                resize=input_frame_size,
                                target_frames=vid_len,
                                frame_diff_interval = frame_diff_interval,
                                dataset=dataset)

test_generator = DataGenerator(directory='{}/processed/test'.format(dataset),
                               batch_size=batch_size,
                               data_augmentation=False,
                               shuffle=False,
                               one_hot=False,
                               sample=False,
                               resize=input_frame_size,
                               target_frames=vid_len,
                               frame_diff_interval = frame_diff_interval,
                               dataset=dataset)

#--------------------------------------------------

print('> getting the model from...', bestValPath)
model = load_model(bestValPath, custom_objects={
                      'SepConvLSTM2D': SepConvLSTM2D})

#--------------------------------------------------

SavePath = '/gdrive/My Drive/THESIS/Data/results/' + str(dataset)+'/'

#--------------------------------------------------

train_results = model.evaluate(
    steps = len(train_generator),
    x=train_generator,
    verbose=1,
    workers=8,
    max_queue_size=8,
    use_multiprocessing=False,
)

test_results = model.evaluate(
    steps = len(test_generator),
    x=test_generator,
    verbose=1,
    workers=8,
    max_queue_size=8,
    use_multiprocessing=False,
)

#----------------------------------------------------------

save_as_csv(train_results, SavePath, 'train_results.csv')
save_as_csv(test_results, SavePath, 'test_resuls.csv')

#----------------------------------------------------------