import os
os.environ['PYTHONHASHSEED'] = '42'
from numpy.random import seed, shuffle
from random import seed as rseed
from tensorflow.random import set_seed
seed(42)
rseed(42)
set_seed(42)
import random
import pickle
import shutil
import models
from utils import *
from dataGenerator import *
from datasetProcess import *
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, ModelCheckpoint,LearningRateScheduler
from tensorflow.python.keras import backend as K
import pandas as pd
import argparse

def train(args):

    mode = args.mode # ["both","only_frames","only_differences"]

    if args.fusionType != 'C':
        if args.mode != 'both':
            print("Only Concat fusion supports one stream versions. Changing mode to /'both/'...")
            mode = "both"
        if args.lstmType == '3dconvblock':
            raise Exception('3dconvblock instead of lstm is only available for fusionType C ! aborting execution...')

    if args.fusionType == 'C':
        model_function = models.getProposedModelC
    elif args.fusionType == 'A':
        model_function = models.getProposedModelA
    elif args.fusionType == 'M':
        model_function = models.getProposedModelM

    dataset = args.dataset # ['rwf2000','movies','hockey']
    dataset_videos = {'hockey':'raw_videos/HockeyFights','movies':'raw_videos/movies'}

    if dataset == "rwf2000":
        initial_learning_rate = 4e-04
    elif dataset == "hockey":
        initial_learning_rate = 1e-06 
    elif dataset == "movies":
        initial_learning_rate = 1e-05 

    batch_size = args.batchSize

    vid_len = args.vidLen  # 32
    if dataset == "rwf2000":
        dataset_frame_size = 320
    else:
        dataset_frame_size = 224
    frame_diff_interval = 1
    input_frame_size = 224

    lstm_type = args.lstmType # attensepconv

    crop_dark = {
        'hockey' : (16,45),
        'movies' : (18,48),
        'rwf2000': (0,0)
    }

    #---------------------------------------------------

    epochs = args.numEpochs

    preprocess_data = args.preprocessData

    create_new_model = ( not args.resume )

    save_path = args.savePath

    resume_path = args.resumePath

    background_suppress = args.noBackgroundSuppression

    if resume_path == "NOT_SET":
        currentModelPath =  os.path.join(save_path , str(dataset) + '_currentModel')
    else:
        currentModelPath = resume_path

    bestValPath =  os.path.join(save_path, str(dataset) + '_best_val_acc_Model')  

    rwfPretrainedPath = args.rwfPretrainedPath
    if rwfPretrainedPath == "NOT_SET":
        
        if lstm_type == "sepconv":
            ###########################
            # rwfPretrainedPath contains path to the model which is already trained on rwf2000 dataset. It is used to initialize training on hockey or movies dataset
            # get this model from the trained_models google drive folder that I provided in readme 
            ###########################
            rwfPretrainedPath = "./trained_models/rwf2000_model/sepconvlstm-M/model/rwf2000_model"   # if you are using M model
        else:
            pass
        

    resume_learning_rate = 5e-05   

    cnn_trainable = True  

    one_hot = False

    loss = 'binary_crossentropy'

    #----------------------------------------------------

    if preprocess_data:

        if dataset == 'rwf2000':
            os.mkdir(os.path.join(dataset, 'processed'))
            convert_dataset_to_npy(src='{}/RWF-2000'.format(dataset), dest='{}/processed'.format(
                dataset), crop_x_y=None, target_frames=vid_len, frame_size= dataset_frame_size)
        else:
            if os.path.exists('{}'.format(dataset)):
                shutil.rmtree('{}'.format(dataset))
            split = train_test_split(dataset_name=dataset,source=dataset_videos[dataset])
            os.mkdir(dataset)
            os.mkdir(os.path.join(dataset,'videos'))
            move_train_test(dest='{}/videos'.format(dataset),data=split)
            os.mkdir(os.path.join(dataset,'processed'))
            convert_dataset_to_npy(src='{}/videos'.format(dataset),dest='{}/processed'.format(dataset), crop_x_y=crop_dark[dataset], target_frames=vid_len, frame_size= dataset_frame_size )

    train_generator = DataGenerator(directory = '{}/processed/train'.format(dataset),
                                    batch_size = batch_size,
                                    data_augmentation = True,
                                    shuffle = True,
                                    one_hot = one_hot,
                                    sample = False,
                                    resize = input_frame_size,
                                    background_suppress = background_suppress,
                                    target_frames = vid_len,
                                    dataset = dataset,
                                    mode = mode)

    test_generator = DataGenerator(directory = '{}/processed/test'.format(dataset),
                                    batch_size = batch_size,
                                    data_augmentation = False,
                                    shuffle = False,
                                    one_hot = one_hot,
                                    sample = False,
                                    resize = input_frame_size,
                                    background_suppress = background_suppress,
                                    target_frames = vid_len,
                                    dataset = dataset,
                                    mode = mode)

    #--------------------------------------------------

    print('> cnn_trainable : ',cnn_trainable)
    if create_new_model:
        print('> creating new model...')
        model = model_function(size=input_frame_size, seq_len=vid_len,cnn_trainable=cnn_trainable, frame_diff_interval = frame_diff_interval, mode=mode, lstm_type=lstm_type)
        if dataset == "hockey" or dataset == "movies":
            print('> loading weights pretrained on rwf dataset from', rwfPretrainedPath)
            model.load_weights(rwfPretrainedPath)
        optimizer = Adam(lr=initial_learning_rate, amsgrad=True)
        model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
        print('> new model created')    
    else:
        print('> getting the model from...', currentModelPath)  
        if dataset == 'rwf2000':
            model =  model_function(size=input_frame_size, seq_len=vid_len,cnn_trainable=cnn_trainable, frame_diff_interval = frame_diff_interval, mode=mode, lstm_type=lstm_type)
            optimizer = Adam(lr=resume_learning_rate, amsgrad=True)
            model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
            model.load_weights(f'{currentModelPath}')
        elif  dataset == "hockey" or dataset == "movies":
            model =  model_function(size=input_frame_size, seq_len=vid_len,cnn_trainable=cnn_trainable, frame_diff_interval = frame_diff_interval, mode=mode, lstm_type=lstm_type)
            optimizer = Adam(lr=initial_learning_rate, amsgrad=True)
            model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
            model.load_weights(f'{currentModelPath}')           

    print('> Summary of the model : ')
    model.summary(line_length=140)
    print('> Optimizer : ', model.optimizer.get_config())

    dot_img_file = 'model_architecture.png'
    print('> plotting the model architecture and saving at ', dot_img_file)
    plot_model(model, to_file=dot_img_file, show_shapes=True)

    #--------------------------------------------------

    modelcheckpoint = ModelCheckpoint(
        currentModelPath, monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', save_freq='epoch')
        
    modelcheckpointVal = ModelCheckpoint(
        bestValPath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')

    historySavePath = os.path.join(save_path, 'results', str(dataset))
    save_training_history = SaveTrainingCurves(save_path = historySavePath)

    callback_list = [
                    modelcheckpoint,
                    modelcheckpointVal,
                    save_training_history
                    ]
                    
    callback_list.append(LearningRateScheduler(lr_scheduler, verbose = 0))
                    
    #--------------------------------------------------

    model.fit(
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--numEpochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--vidLen', type=int, default=32, help='Number of frames in a clip')
    parser.add_argument('--batchSize', type=int, default=4, help='Training batch size')
    parser.add_argument('--resume', help='whether training should resume from the previous checkpoint',action='store_true')
    parser.add_argument('--noBackgroundSuppression', help='whether to use background suppression on frames',action='store_false')
    parser.add_argument('--preprocessData', help='whether need to preprocess data ( make npy file from video clips )',action='store_true')
    parser.add_argument('--mode', type=str, default='both', help='model type - both, only_frames, only_differences', choices=['both', 'only_frames', 'only_differences']) 
    parser.add_argument('--dataset', type=str, default='rwf2000', help='dataset - rwf2000, movies, hockey', choices=['rwf2000','movies','hockey']) 
    parser.add_argument('--lstmType', type=str, default='sepconv', help='lstm - conv, sepconv, asepconv, 3dconvblock(use 3dconvblock instead of lstm)', choices=['sepconv','asepconv', 'conv', '3dconvblock'])
    parser.add_argument('--fusionType', type=str, default='C', help='fusion type - A for add, M for multiply, C for concat', choices=['C','A','M']) 
    parser.add_argument('--savePath', type=str, default='/gdrive/My Drive/THESIS/Data', help='folder path to save the models')
    parser.add_argument('--rwfPretrainedPath', type=str, default='NOT_SET', help='path to the weights pretrained on rwf dataset')
    parser.add_argument('--resumePath', type=str, default='NOT_SET', help='path to the weights for resuming from previous checkpoint')
    parser.add_argument('--resumeLearningRate', type=float, default=5e-05, help='learning rate to resume training from')
    args = parser.parse_args()
    train(args)

main()
