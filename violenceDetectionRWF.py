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
from tensorflow.keras.utils import plot_model
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

initial_learning_rate = 4e-04
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

create_new_model = False

bestModelPath = '/gdrive/My Drive/THESIS/Data/' + \
    str(dataset) + '_bestModel.h5'

bestValPath =  '/gdrive/My Drive/THESIS/Data/' + \
    str(dataset) + '_best_val_acc_Model.h5'   

epochs = 5

learning_rate = None   

cnn_trainable = True

###################################################

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
                                target_frames = vid_len,
                                frame_diff_interval = frame_diff_interval,
                                dataset = dataset)

test_generator = DataGenerator(directory='{}/processed/test'.format(dataset),
                               batch_size=batch_size,
                               data_augmentation=False,
                               shuffle=True,
                               one_hot=False,
                               sample=False,
                               resize=input_frame_size,
                               target_frames = vid_len,
                               frame_diff_interval = frame_diff_interval,
                               dataset = dataset)

#--------------------------------------------------

print('> cnn_trainable : ',cnn_trainable)
if create_new_model:
    print('> creating new model...')
    model =  sepConvLstmNet.getModel(size=input_frame_size, seq_len=vid_len,cnn_trainable=cnn_trainable, frame_diff_interval = frame_diff_interval)
    optimizer = Adam(lr=initial_learning_rate, amsgrad=True)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
    print('> Dropout on FC layer : ', model.layers[-2].rate)
    print('> new model created')
else:
    print('> getting the model from...', bestModelPath)
    model = load_model(bestModelPath, custom_objects={
                      'SepConvLSTM2D': SepConvLSTM2D})
    # freezing/unfreezing the CNN
    # for layer in model.layers[1].layer.layers: 
    #    layer.trainable = cnn_trainable 
    if learning_rate is not None:
        K.set_value(model.optimizer.lr, learning_rate)  
    # recompiling the model          
    # model.compile(optimizer=model.optimizer, loss='binary_crossentropy', metrics=['acc'])
    print('> Dropout on FC layer : ', model.layers[-2].rate)

print('> Summary of the model : ')
model.summary(line_length=140)
print('> Optimizer : ', model.optimizer.get_config())

dot_img_file = 'model_architecture.png'
print('> plotting the model architecture and saving at ', dot_img_file)
plot_model(model, to_file=dot_img_file, show_shapes=True)

#--------------------------------------------------

modelcheckpoint = ModelCheckpoint(
    bestModelPath, monitor='loss', verbose=0, save_best_only=False, mode='auto', save_freq='epoch')
    
modelcheckpointVal = ModelCheckpoint(
    bestValPath, monitor='val_acc', verbose=0, save_best_only=True, mode='auto', save_freq='epoch')

historySavePath = '/gdrive/My Drive/THESIS/Data/results/' + str(dataset)+'/'
save_training_history = SaveTrainingCurves(save_path = historySavePath)

#--------------------------------------------------

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
    callbacks=[
                LearningRateScheduler(lr_scheduler, verbose = 0),
                modelcheckpoint,
                modelcheckpointVal,
                save_training_history
              ]
)

#----------------------------------------------------------

