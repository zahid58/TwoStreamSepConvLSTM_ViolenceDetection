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

initial_learning_rate = 2e-04
dataset = 'movies'
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

###################################################

split_number = 1

preprocess_data = True
create_new_model = True

if not os.path.exists('/gdrive/My Drive/THESIS/Data/models'):
    os.makedirs('/gdrive/My Drive/THESIS/Data/models')

bestModelPath = '/gdrive/My Drive/THESIS/Data/models/' + \
    str(dataset) + '_bestModel.h5'

bestValPath =  '/gdrive/My Drive/THESIS/Data/models/' + \
    str(dataset) + '_best_val_acc_Model.h5'   

epochs = 50

learning_rate = None   

cnn_trainable = True

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
                                    data_augmentation=True,
                                    shuffle=True,
                                    one_hot = False,
                                    sample = False,
                                    resize = input_frame_size,
                                    target_frames=vid_len,
                                    dataset=dataset)

test_generator = DataGenerator(directory='{}/processed/test'.format(dataset),
                                    batch_size=batch_size, 
                                    data_augmentation=False,
                                    shuffle=True,
                                    one_hot=False,
                                    sample = False,
                                    resize = input_frame_size,
                                    target_frames=vid_len,
                                    dataset=dataset)

#--------------------------------------------------
print('> cnn_trainable : ',cnn_trainable)
if create_new_model:
    print('> creating new model...')
    model =  sepConvLstmNet.getModel(size=input_frame_size, seq_len=vid_len,cnn_trainable=cnn_trainable)
    print('new model created')
    rwfPretrainedPath = '/gdrive/My Drive/THESIS/Data/models/rwf2000_best_val_acc_Model.h5'
    print('> loading weights pretrained on rwf dataset from', rwfPretrainedPath)
    model.load_weights(rwfPretrainedPath)
    print('pretrained weights loaded !')
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

dot_img_file = '/gdrive/My Drive/THESIS/Data/results/' + str(dataset) + '/model_architecture.png'
print('> plotting the model architecture and saving at ', dot_img_file)
#plot_model(model, to_file='model_architecture.png', show_shapes=True) #local
#plot_model(model, to_file=dot_img_file, show_shapes=True) #drive

#--------------------------------------------------

modelcheckpoint = ModelCheckpoint(
    bestModelPath, monitor='loss', verbose=0, save_best_only=False, mode='auto', save_freq='epoch')
    
modelcheckpointVal = ModelCheckpoint(
    bestValPath, monitor='val_acc', verbose=0, save_best_only=True, mode='auto', save_freq='epoch')

historySavePath = '/gdrive/My Drive/THESIS/Data/results/' + str(dataset)+'/'
save_training_history = SaveTrainingCurves(save_path = historySavePath, split_num = split_number)

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
