import random
from customLayers import SepConvLSTM2D
import pickle
import shutil
import sepConvLstmNet
from utils import *
from dataGenerator import *
from datasetProcess import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, ModelCheckpoint,LearningRateScheduler
from tensorflow.python.keras import backend as K
from tensorflow.random import set_seed
import os
import pandas as pd
from numpy.random import seed, shuffle
seed(42)
random.seed(42)
set_seed(42)


#-----------------------------------

dataset = 'movies'  # 'hockey'
dataset_videos = {'hockey':'raw_videos/HockeyFights','movies':'raw_videos/movies'}
crop_dark = {
    'hockey' : (16,45),
    'movies' : (18,48),
    'rwf2000': (0,0)
}
batch_size = 4
vid_len = 32
dataset_frame_size = 224
input_frame_size = 224
frame_diff_interval = 1
###################################################

split_number = 1

preprocess_data = False

if not os.path.exists('/gdrive/My Drive/THESIS/Data/models'):
    os.makedirs('/gdrive/My Drive/THESIS/Data/models')

currentModelPath = '/gdrive/My Drive/THESIS/Data/models/' + \
    str(dataset) + '_currentModel.h5'

bestValPath =  '/gdrive/My Drive/THESIS/Data/models/' + \
    str(dataset) + '_best_val_acc_Model.h5'   


###################################################

if preprocess_data:
    if os.path.exists('{}'.format(dataset)):
        shutil.rmtree('{}'.format(dataset))
    splits = five_fold_split(dataset_name=dataset,source=dataset_videos[dataset])
    os.mkdir(dataset)
    os.mkdir(os.path.join(dataset,'videos'))
    move_train_test(dest='{}/videos'.format(dataset),data=splits[split_number-1])
    os.mkdir(os.path.join(dataset,'processed'))
    convert_dataset_to_npy(src='{}/videos'.format(dataset),dest='{}/processed'.format(dataset), crop_x_y=crop_dark[dataset], target_frames=vid_len, frame_size= dataset_frame_size )

train_generator = DataGenerator(directory='{}/processed/train'.format(dataset), 
                                    batch_size=batch_size, 
                                    data_augmentation=False,
                                    shuffle=False,
                                    one_hot = False,
                                    sample = False,
                                    resize = input_frame_size,
                                    target_frames=vid_len,
                                    frame_diff_interval = frame_diff_interval,
                                    dataset=dataset)

test_generator = DataGenerator(directory='{}/processed/test'.format(dataset),
                                    batch_size=batch_size, 
                                    data_augmentation=False,
                                    shuffle=False,
                                    one_hot=False,
                                    sample = False,
                                    resize = input_frame_size,
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