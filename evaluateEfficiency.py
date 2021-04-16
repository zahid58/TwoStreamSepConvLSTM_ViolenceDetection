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
from tensorflow.python.keras import backend as K
import pandas as pd
import argparse

def evaluateEfficiency(args):

    #--------------------------------------------------
    flops, params = get_flops(args)
    print("============================")
    print('fusion:', args.fusionType)
    print('lstm type:', args.lstmType)
    print('input mode:', args.mode)
    print('----------------------------')
    print('FLOPs:',flops)
    print('Parameters:',params)
    print('============================')
    #---------------------------------------------------


def get_flops(args, save_results_to_file=False):

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

    vid_len = args.vidLen  # 32

    frame_diff_interval = 1
    input_frame_size = 224

    lstm_type = args.lstmType

    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():

            if args.flowGatedNet:
                model = models.getFlowGatedNet()
            else:
                model = model_function(size=input_frame_size, seq_len=vid_len,cnn_trainable=True, frame_diff_interval = frame_diff_interval, mode=mode, lstm_type=lstm_type)
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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vidLen', type=int, default=32, help='Number of frames in a clip')
    parser.add_argument('--batchSize', type=int, default=4, help='Training batch size')
    parser.add_argument('--preprocessData', help='whether need to preprocess data ( make npy file from video clips )',action='store_true')
    parser.add_argument('--mode', type=str, default='both', help='model type - both, only_frames, only_difference', choices=['both', 'only_frames', 'only_difference']) 
    parser.add_argument('--dataset', type=str, default='rwf2000', help='dataset - rwf2000, movies, hockey', choices=['rwf2000','movies','hockey']) 
    parser.add_argument('--lstmType', type=str, default='sepconv', help='lstm - sepconv, asepconv', choices=['sepconv','asepconv']) 
    parser.add_argument('--weightsPath', type=str, default='NOT_SET', help='path to the weights pretrained on rwf dataset')
    parser.add_argument('--fusionType', type=str, default='concat', help='fusion type - A for add, M for multiply, C for concat', choices=['C','A','M']) 
    parser.add_argument('--flowGatedNet', help='measure the efficiency of FlowGatedNet by Ming et. al.',action='store_true')
    args = parser.parse_args()
    evaluateEfficiency(args)

main()


#--------------------------------------------------
