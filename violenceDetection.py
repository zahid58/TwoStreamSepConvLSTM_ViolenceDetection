import os
import pandas as pd
from numpy.random import seed, shuffle
seed(42)
import random 
random.seed(42)
from tensorflow.random import set_seed
set_seed(42)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import load_model
from datasetProcess import *
from dataGenerator import *
from utils import *
import sepConvLstmNet
import shutil
import pickle
from customLayers import SepConvLSTM2D

dataset = 'hockey'
dataset_videos = {'hockey':'raw_videos/HockeyFights','movies':'raw_videos/movies'}
crop_dark = {
    'hockey' : (16,45),
    'movies' : (18,48)
}

batch_size = 4   #4
vid_len = 20   #10
frame_size = 224

preprocess_data = False  #######
split_number = 1   #[1,2,3,4,5,6]
create_new_model = True  #######
bestModelPath = '/gdrive/My Drive/THESIS/Data/' + str(dataset) + '_bestModel.h5'
epochs = 10

if preprocess_data:
    splits = five_fold_split(dataset_name=dataset,source=dataset_videos[dataset])
    os.mkdir(dataset)
    os.mkdir(os.path.join(dataset,'videos'))
    move_train_test(dest='{}/videos'.format(dataset),data=splits[split_number-1])
    os.mkdir(os.path.join(dataset,'processed'))
    convert_dataset_to_npy(src='{}/videos'.format(dataset),dest='{}/processed'.format(dataset),crop_x_y=crop_dark[dataset])


train_generator = DataGenerator(directory='{}/processed/train'.format(dataset), 
                                    batch_size=batch_size, 
                                    data_augmentation=True,
                                    shuffle=False,
                                    one_hot = False,
                                    target_frames=vid_len)

test_generator = DataGenerator(directory='{}/processed/test'.format(dataset),
                                    batch_size=batch_size, 
                                    data_augmentation=False,
                                    shuffle=False,
                                    one_hot=False,
                                    target_frames=vid_len)


model = None
if create_new_model:
    print('creating new model...')
    model = sepConvLstmNet.getModel(size=frame_size, seq_len=vid_len , cnn_weight = 'imagenet')
    print('new model created')
else:
    print('getting the model from ',bestModelPath)
    model = load_model(bestModelPath,custom_objects={'SepConvLSTM2D':SepConvLSTM2D})
    print('got the model')
print(model.summary())

optimizer = Adam(lr=5e-05)
model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['acc'])
modelcheckpoint = ModelCheckpoint(bestModelPath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', save_freq='epoch')    

history = model.fit(
        steps_per_epoch=len(train_generator),
        x=train_generator,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=len(test_generator),
        verbose = 1,
        workers = 8,
        max_queue_size= 8,
        use_multiprocessing = False,
        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001, patience=15 ),
                   ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-8, verbose=1),
                   modelcheckpoint
                   ]
        )

history_to_save = history.history
savePath = '/gdrive/My Drive/THESIS/Data/' + str(dataset)
save_plot_history(history=history_to_save, save_path= savePath)

historyFile = savePath + '_history.pickle'
try:
    file_ = open(historyFile, 'wb')
    pickle.dump(history_to_save, file_)
    print('saved', historyFile)
except Exception as e:
    print(e)