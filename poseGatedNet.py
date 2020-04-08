from keras import Input
from keras.callbacks import Callback
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, ConvLSTM2D, Reshape, BatchNormalization, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
import sys
from keras.applications import Xception, ResNet50, InceptionV3, MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Multiply, MaxPooling2D,Concatenate,Add
from keras.models import Model
from keras.optimizers import Adam

def getModel(size=224, seq_len=20 , cnn_weight=None, lstm_conf=None ):
    """parameter:
    size = height/width of each frame,
    seq_len = number of frames in each sequence,
    cnn_weight= None or 'imagenet',
    lstm_conf = a rnn layer object 
    """
    image_input = Input(shape=(seq_len, size, size, 3))
    pose_input = Input(shape=(seq_len, size, size, 3))
    
    cnn = MobileNetV2(weights= cnn_weight, include_top=False,input_shape =(size, size, 3))

    for layer in cnn.layers:
        layer.trainable = True

    pose_cnn = MobileNetV2(weights = None, include_top = False, input_shape = (size, size, 3))

    for layer in pose_cnn.layers:
        layer.trainable = True
    
    pose_cnn = Model(pose_cnn.input, pose_cnn.layers[-2].output)
    
    pose_cnn = TimeDistributed(pose_cnn)(pose_input)
    pose_cnn = TimeDistributed(Activation('sigmoid'))(pose_cnn)
    
    cnn = TimeDistributed(cnn)(image_input)

    multiplied = Multiply()([cnn, pose_cnn])
    #multiplied = TimeDistributed(MaxPooling2D(pool_size=(2,2))) (multiplied)
    
    lstm = lstm_conf[0](**lstm_conf[1])(multiplied)
    lstm = MaxPooling2D(pool_size=(2, 2))(lstm)   
    flat = Flatten()(lstm)
    x = BatchNormalization()(flat)
    
    x = Dense(1000,activation = 'relu')(x)
    x = Dropout(.2)(x)
    
    x = Dense(256,activation='relu')(x)
    x = Dropout(.1)(x)

    x = Dense(10,activation='relu')(x)
    x = Dropout(.1)(x)

    activation = 'sigmoid'
    loss_func = 'binary_crossentropy'
    classes = 1

    if classes > 1:
        activation = 'softmax'
        loss_func = 'categorical_crossentropy'
    predictions = Dense(classes,  activation=activation)(x)

    model = Model(inputs=[image_input, pose_input], outputs=predictions)
    
    print(model.summary())

    return model