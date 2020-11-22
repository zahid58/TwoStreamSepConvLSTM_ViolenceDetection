from tensorflow.keras import Input
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, ConvLSTM2D, Reshape, BatchNormalization, Activation, Conv2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.applications import MobileNetV2, VGG16
from tensorflow.keras.layers import ELU, ReLU, LeakyReLU, Lambda, Dense, Bidirectional, Conv3D, GlobalAveragePooling2D, Multiply, MaxPooling3D, MaxPooling2D, Concatenate, Add, AveragePooling2D 
from tensorflow.keras.initializers import glorot_uniform, he_normal
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras import backend as K
import numpy as np
from tensorflow.keras.layers import UpSampling2D, Concatenate
from mobilenet_v3 import MobileNetV3Small
from yolo_layers import tiny_yolo3lite_predictions, tiny_yolo3_ultralite_predictions

def get_yolo3_model(model_type, input_shape, weights_path):
    if model_type == "tiny_yolo3_mobilenetv3_small":
        num_anchors=6; num_classes=1
        model = tiny_yolo3lite_mobilenetv3small_body(input_shape, num_anchors//2, num_classes)
    elif model_type == "tiny_yolo3_mobilenetv3small_ultralite":
        num_anchors=6; num_classes=1
        model = tiny_yolo3_ultralite_mobilenetv3small_body(input_shape, num_anchors//2, num_classes)           
    model.load_weights(weights_path)
    return model


def tiny_yolo3lite_mobilenetv3small_body(input_shape, num_anchors, num_classes, alpha=1.0):
    '''Create Tiny YOLO_v3 Lite MobileNetV3Small model CNN body in keras.'''
    mobilenetv3small = MobileNetV3Small(input_shape=input_shape, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv3small.layers)))

    # input: 416 x 416 x 3
    # activation_31(layer 165, final feature map): 13 x 13 x (576*alpha)
    # expanded_conv_10/Add(layer 162, end of block10): 13 x 13 x (96*alpha)

    # activation_22(layer 117, middle in block8) : 26 x 26 x (288*alpha)
    # expanded_conv_7/Add(layer 114, end of block7) : 26 x 26 x (48*alpha)

    # activation_7(layer 38, middle in block3) : 52 x 52 x (96*alpha)
    # expanded_conv_2/Add(layer 35, end of block2): 52 x 52 x (24*alpha)

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1 :13 x 13 x (576*alpha)
    f1 = mobilenetv3small.layers[165].output
    # f2: 26 x 26 x (288*alpha)
    f2 = mobilenetv3small.layers[117].output

    f1_channel_num = int(576*alpha)
    f2_channel_num = int(288*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512

    y1, y2 = tiny_yolo3lite_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(mobilenetv3small.input, [y1,y2])


def tiny_yolo3_ultralite_mobilenetv3small_body(input_shape, num_anchors, num_classes, alpha=1.0):
    '''Create Tiny YOLO_v3 Ultra-Lite MobileNetV3Small model CNN body in keras.'''
    mobilenetv3small = MobileNetV3Small(input_shape=input_shape, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenetv3small.layers)))

    # input: 416 x 416 x 3
    # activation_31(layer 165, final feature map): 13 x 13 x (576*alpha)
    # expanded_conv_10/Add(layer 162, end of block10): 13 x 13 x (96*alpha)

    # activation_22(layer 117, middle in block8) : 26 x 26 x (288*alpha)
    # expanded_conv_7/Add(layer 114, end of block7) : 26 x 26 x (48*alpha)

    # activation_7(layer 38, middle in block3) : 52 x 52 x (96*alpha)
    # expanded_conv_2/Add(layer 35, end of block2): 52 x 52 x (24*alpha)

    # NOTE: activation layer name may different for TF1.x/2.x, so we
    # use index to fetch layer
    # f1 :13 x 13 x (576*alpha)
    f1 = mobilenetv3small.layers[165].output
    # f2: 26 x 26 x (288*alpha)
    f2 = mobilenetv3small.layers[117].output

    f1_channel_num = int(576*alpha)
    f2_channel_num = int(288*alpha)
    #f1_channel_num = 1024
    #f2_channel_num = 512

    y1, y2 = tiny_yolo3_ultralite_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(mobilenetv3small.input, [y1,y2])
