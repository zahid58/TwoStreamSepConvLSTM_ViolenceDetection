import os
os.environ['PYTHONHASHSEED'] = '42'
from numpy.random import seed, shuffle
from random import seed as rseed
from tensorflow.random import set_seed
seed(42)
rseed(42)
set_seed(42)
import random
from sep_conv_rnn import SepConvLSTM2D
import pickle
import shutil
import cnn_lstm_models
import rwfRGBonly
from utils import *
from dataGenerator import *
from datasetProcess import *
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, ModelCheckpoint,LearningRateScheduler
from tensorflow.python.keras import backend as K
import pandas as pd


#-----------------------------------

model_type = "proposed" # [ "proposed", "convlstm", "biconvlstm"]
mode = "both" # ["both","only_frames","only_differneces"]

dataset = 'rwf2000' 
dataset_videos = {'hockey':'raw_videos/HockeyFights','movies':'raw_videos/movies'}

if model_type == "proposed":
    if dataset == "rwf2000":
        initial_learning_rate = 4e-04
    else:
        initial_learning_rate = 1e-06
elif model_type == "biconvlstm":
    initial_learning_rate = 1e-06
elif model_type == "convlstm":
    initial_learning_rate = 1e-05   

batch_size = 4
if model_type == "biconvlstm":
    batch_size = 2
vid_len = 32  # 32
dataset_frame_size = 320
input_frame_size = 224
frame_diff_interval = 1
if model_type == "convlstm" or model_type == "biconvlstm":
    mode = "only_differences"  
lstm_type = 'sepconv' # attensepconv

crop_dark = {
    'hockey' : (16,45),
    'movies' : (18,48),
    'rwf2000': (0,0),
    'surv': (0,0)
}


#---------------------------------------------------

epochs = 50

preprocess_data = False

create_new_model = False

currentModelPath = '/gdrive/My Drive/THESIS/Data/' + \
    str(dataset) + '_currentModel'

bestValPath =  '/gdrive/My Drive/THESIS/Data/' + \
    str(dataset) + '_best_val_acc_Model'   

rwfPretrainedPath = 'NOT_SET'

learning_rate = None   

cnn_trainable = True  

loss = 'categorical_crossentropy'

one_hot = True
if model_type == "proposed":
    one_hot = False
    loss = 'binary_crossentropy'

#---------------------------------------------------

if preprocess_data:

    if dataset == 'rwf2000':
        os.mkdir(os.path.join(dataset, 'processed'))
        convert_dataset_to_npy(src='{}/RWF-2000'.format(dataset), dest='{}/processed'.format(
            dataset), crop_x_y=None, target_frames=vid_len, frame_size= dataset_frame_size)

    elif dataset == 'surv':
        if os.path.exists('{}'.format(dataset)):
            shutil.rmtree('{}'.format(dataset))
        splits = five_fold_split(dataset_name=dataset,source=dataset_videos[dataset])
        os.mkdir(dataset)
        os.mkdir(os.path.join(dataset,'videos'))
        split_number = 1
        move_train_test(dest='{}/videos'.format(dataset),data=splits[split_number-1])
        os.mkdir(os.path.join(dataset,'processed'))
        convert_dataset_to_npy(src='{}/videos'.format(dataset),dest='{}/processed'.format(dataset), crop_x_y=crop_dark[dataset], target_frames=vid_len, frame_size= dataset_frame_size )



train_generator = DataGenerator(directory='{}/processed/train'.format(dataset),
                                batch_size=batch_size,
                                data_augmentation=True,
                                shuffle=True,
                                one_hot=one_hot,
                                sample=False,
                                resize=input_frame_size,
                                target_frames = vid_len,
                                frame_diff_interval = frame_diff_interval,
                                dataset = dataset,
                                mode = mode)

test_generator = DataGenerator(directory='{}/processed/test'.format(dataset),
                               batch_size=batch_size,
                               data_augmentation=False,
                               shuffle=True,
                               one_hot=one_hot,
                               sample=False,
                               resize=input_frame_size,
                               target_frames = vid_len,
                               frame_diff_interval = frame_diff_interval,
                               dataset = dataset,
                               mode = mode)

#--------------------------------------------------

print('> cnn_trainable : ',cnn_trainable)
if create_new_model:
    if model_type == "proposed":
        print('> creating new model...', model_type)
        model = cnn_lstm_models.getProposedModel(size=input_frame_size, seq_len=vid_len,cnn_trainable=cnn_trainable, frame_diff_interval = frame_diff_interval, mode="both", lstm_type=lstm_type)
        if dataset == "hockey" or dataset == "movies":
                print('> loading weights pretrained on rwf dataset from', rwfPretrainedPath)
                model.load_weights(rwfPretrainedPath)
        optimizer = Adam(lr=initial_learning_rate, amsgrad=True)
        model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
        print('> new model created')    
else:
    print('> getting the model from...', currentModelPath)  
    if model_type == "proposed":
        model =  cnn_lstm_models.getProposedModel(size=input_frame_size, seq_len=vid_len,cnn_trainable=cnn_trainable, frame_diff_interval = frame_diff_interval, mode="both", lstm_type=lstm_type)
        optimizer = Adam(lr=5e-05, amsgrad=True)
        model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
        model.load_weights(currentModelPath)
    else:
        raise Exception("NOT DEFINED WHAT TO DO!")
        #model = load_model(currentModelPath)
    # if learning_rate is not None:
    #     K.set_value(model.optimizer.lr, learning_rate)  

print('> Summary of the model : ')
model.summary(line_length=140)
print('> Optimizer : ', model.optimizer.get_config())

dot_img_file = 'model_architecture.png'
print('> plotting the model architecture and saving at ', dot_img_file)
plot_model(model, to_file=dot_img_file, show_shapes=True)


#--------------------------------------------------

modelcheckpoint = ModelCheckpoint(
    currentModelPath, monitor='loss', verbose=0, save_best_only=False, save_weights_only = True, mode='auto', save_freq='epoch')
    
modelcheckpointVal = ModelCheckpoint(
    bestValPath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only = True, mode='auto', save_freq='epoch')

historySavePath = '/gdrive/My Drive/THESIS/Data/results/' + str(dataset)+'/'
save_training_history = SaveTrainingCurves(save_path = historySavePath)

callback_list = [
                modelcheckpoint,
                modelcheckpointVal,
                save_training_history
                ]
                
if model_type == "proposed":
    callback_list.append(LearningRateScheduler(lr_scheduler, verbose = 0))
                
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
    callbacks= callback_list
)

#---------------------------------------------------
