import os
os.environ['PYTHONHASHSEED'] = '42'
from numpy.random import seed, shuffle
from random import seed as rseed
from tensorflow.random import set_seed
seed(42)
rseed(42)
set_seed(42)
import random
from customLayers import SepConvLSTM2D
import pickle
import shutil
import sepConvLstmNet
import cnn_lstm_models
import rwfRGBonly
from utils import *
from dataGenerator import *
from datasetProcess import *
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, ModelCheckpoint,LearningRateScheduler
from tensorflow.python.keras import backend as K
import pandas as pd
#-----------------------------------

model_type = "proposed" # [ "proposed", "convlstm", "biconvlstm"]
mode = "both" # ["both","only_frames","only_differneces"]
initial_learning_rate = 4e-04
if model_type == "biconvlstm":
    initial_learning_rate = 1e-06
elif model_type == "convlstm":
    initial_learning_rate = 1e-05    

batch_size = 4
if model_type == "biconvlstm":
    batch_size = 2
vid_len = 32
dataset_frame_size = 320
input_frame_size = 224
frame_diff_interval = 1
if model_type == "convlstm" or model_type == "biconvlstm":
    mode = "only_differences"  

lstm_type = "sepconv"  # conv

#---------------------------------------------------

learning_rate = None   

cnn_trainable = True

loss = 'categorical_crossentropy'
one_hot = True
if model_type == "proposed":
    one_hot = False
    loss = 'binary_crossentropy'

#--------------------------------------------------

def get_flops(save_results_to_file=False):

    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            
            print('> creating new model...', model_type)
            if model_type == "proposed":
                model =  cnn_lstm_models.getProposedModel(size=input_frame_size, seq_len=vid_len,cnn_trainable=cnn_trainable, frame_diff_interval = frame_diff_interval, mode=mode,lstm_type=lstm_type)
            elif model_type == "convlstm":
                model =  cnn_lstm_models.getConvLSTM(size=input_frame_size, seq_len=vid_len,cnn_trainable=cnn_trainable, frame_diff_interval = frame_diff_interval, mode = mode)
            elif model_type == "biconvlstm":
                model =  cnn_lstm_models.getBiConvLSTM(size=input_frame_size, seq_len=vid_len,cnn_trainable=cnn_trainable, frame_diff_interval = frame_diff_interval, mode = mode)
            params = model.count_params()
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            
            if save_results_to_file:
                # Optional: save printed results to file
                # flops_log_path = os.path.join(tempfile.gettempdir(), 'tf_flops_log.txt')
                # opts['output'] = 'file:outfile={}'.format(flops_log_path)
                pass

            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)

    tf.compat.v1.reset_default_graph()

    return flops.total_float_ops, params


flops, params = get_flops()
print("============================")
print('Model:', model_type)
print('batch size:', batch_size)
print('input mode:', mode)
print('----------------------------')
print('FLOPs:',flops)
print('Parameters:',params)
print('============================')
#--------------------------------------------------
