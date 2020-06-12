import random
from customLayers import SepConvLSTM2D
import pickle
import shutil
import sepConvLstmNet
from utils import *
from dataGenerator import *
from datasetProcess import *
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, ModelCheckpoint
from tensorflow.random import set_seed
import os
import pandas as pd
from numpy.random import seed, shuffle
seed(42)
random.seed(42)
set_seed(42)

dataset = 'rwf2000'

crop_dark = {
    'rwf2000': (0, 0),
}

batch_size = 4
vid_len = 32
dataset_frame_size = 320
input_frame_size = 224

preprocess_data = False
create_new_model = True
bestModelPath = '/gdrive/My Drive/THESIS/Data/' + \
    str(dataset) + '_bestModel.h5'
epochs = 50

if preprocess_data:
    os.mkdir(os.path.join(dataset, 'processed'))
    convert_dataset_to_npy(src='{}/RWF-2000'.format(dataset), dest='{}/processed'.format(
        dataset), crop_x_y=None, target_frames=vid_len, frame_size= dataset_frame_size)


train_generator = DataGenerator(directory='{}/processed/train'.format(dataset),
                                batch_size=batch_size,
                                data_augmentation=True,
                                shuffle=True,
                                one_hot=False,
                                sample=False,
                                resize=input_frame_size,
                                target_frames = vid_len)

test_generator = DataGenerator(directory='{}/processed/test'.format(dataset),
                               batch_size=batch_size,
                               data_augmentation=False,
                               shuffle=True,
                               one_hot=False,
                               sample=False,
                               resize=input_frame_size,
                               target_frames = vid_len)

model = None
if create_new_model:
    print('creating new model...')
    model = sepConvLstmNet.getModel(
        size=input_frame_size, seq_len=vid_len, cnn_weight='imagenet')
    print('new model created')
else:
    print('getting the model from ', bestModelPath)
    model = load_model(bestModelPath, custom_objects={
                       'SepConvLSTM2D': SepConvLSTM2D})
    print('got the model')
    print('Dropout : ', model.layers[-2].rate)
print(model.summary())

optimizer = Adam(lr=5e-04)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
modelcheckpoint = ModelCheckpoint(
    bestModelPath, monitor='loss', verbose=0, save_best_only=False, mode='auto', save_freq='epoch')

history = model.fit(
    steps_per_epoch=len(train_generator),
    x=train_generator,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=len(test_generator),
    verbose=1,
    workers=8,
    max_queue_size=8,
    use_multiprocessing=False,
    callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001, patience=65),
               ReduceLROnPlateau(monitor='loss', factor=0.5,
                                 patience=3, min_lr=1e-8, verbose=1),
               modelcheckpoint
               ]
)

history_to_save = history.history
savePath = '/gdrive/My Drive/THESIS/Data/results/' + str(dataset)+'/'
save_plot_history(history=history_to_save, save_path=savePath)
