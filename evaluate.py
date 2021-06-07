import os
os.environ['PYTHONHASHSEED'] = '42'
from numpy.random import seed, shuffle
from random import seed as rseed
from tensorflow.random import set_seed
seed(42)
rseed(42)
set_seed(42)
import tensorflow as tf
import random
import pickle
import shutil
import models
from utils import *
from dataGenerator import *
from datasetProcess import *
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras import backend as K
import pandas as pd
import argparse
from tensorflow.keras.optimizers import RMSprop, Adam

def evaluate(args):

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

    batch_size = args.batchSize

    vid_len = args.vidLen  # 32
    if dataset == "rwf2000":
        dataset_frame_size = 320
    else:
        dataset_frame_size = 224
    frame_diff_interval = 1
    input_frame_size = 224

    lstm_type = args.lstmType

    crop_dark = {
        'hockey' : (16,45),
        'movies' : (18,48),
        'rwf2000': (0,0)
    }

    #---------------------------------------------------

    preprocess_data = args.preprocessData

    weightsPath = args.weightsPath
    if weightsPath == "NOT_SET":
        raise Exception("weights not provided!")

    one_hot = False

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


    # train_generator = DataGenerator(directory = '{}/processed/train'.format(dataset),
    #                                 batch_size = batch_size,
    #                                 data_augmentation = False,
    #                                 shuffle = False,
    #                                 one_hot = one_hot,
    #                                 sample = False,
    #                                 resize = input_frame_size,
    #                                 background_suppress = True,
    #                                 target_frames = vid_len,
    #                                 dataset = dataset,
    #                                 mode = mode)

    test_generator = DataGenerator(directory = '{}/processed/test'.format(dataset),
                                    batch_size = batch_size,
                                    data_augmentation = False,
                                    shuffle = False,
                                    one_hot = one_hot,
                                    sample = False,
                                    resize = input_frame_size,
                                    background_suppress = True,
                                    target_frames = vid_len,
                                    dataset = dataset,
                                    mode = mode)

    #--------------------------------------------------

    print('> getting the model from...', weightsPath)  
    model =  model_function(size=input_frame_size, seq_len=vid_len, frame_diff_interval = frame_diff_interval, mode="both", lstm_type=lstm_type)
    optimizer = Adam(lr=4e-4, amsgrad=True)
    loss = 'binary_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
    model.load_weights(f'{weightsPath}').expect_partial()
    model.trainable = False

    # print('> Summary of the model : ')
    # model.summary(line_length=140)

    # dot_img_file = 'model_architecture.png'
    # print('> plotting the model architecture and saving at ', dot_img_file)
    # plot_model(model, to_file=dot_img_file, show_shapes=True)
                    
    #--------------------------------------------------

    # train_results = model.evaluate(
    #     steps = len(train_generator),
    #     x=train_generator,
    #     verbose=1,
    #     workers=8,
    #     max_queue_size=8,
    #     use_multiprocessing=False,
    # )

    test_results = model.evaluate(
        steps = len(test_generator),
        x=test_generator,
        verbose=1,
        workers=8,
        max_queue_size=8,
        use_multiprocessing=False,
    )
    print("====================")
    print("     Results        ")
    print("====================")
    print("> Test Loss:", test_results[0])
    print("> Test Accuracy:", test_results[1])
    print("====================")
    # save_as_csv(train_results, "", 'train_results.csv')
    save_as_csv(test_results, "", 'test_resuls.csv')

    #---------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vidLen', type=int, default=32, help='Number of frames in a clip')
    parser.add_argument('--batchSize', type=int, default=4, help='Training batch size')
    parser.add_argument('--preprocessData', help='whether need to preprocess data ( make npy file from video clips )',action='store_true')
    parser.add_argument('--mode', type=str, default='both', help='model type - both, only_frames, only_difference', choices=['both', 'only_frames', 'only_difference']) 
    parser.add_argument('--dataset', type=str, default='rwf2000', help='dataset - rwf2000, movies, hockey', choices=['rwf2000','movies','hockey']) 
    parser.add_argument('--lstmType', type=str, default='sepconv', help='lstm - sepconv, asepconv', choices=['sepconv','asepconv']) 
    parser.add_argument('--weightsPath', type=str, default='NOT_SET', help='path to the weights pretrained on rwf dataset')
    parser.add_argument('--fusionType', type=str, default='C', help='fusion type - A for add, M for multiply, C for concat', choices=['C','A','M']) 
    args = parser.parse_args()
    evaluate(args)

main()
