from keras import Input
from keras.callbacks import Callback
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, ConvLSTM2D, Reshape, BatchNormalization, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from collections import deque
import sys
import logging
from keras.applications import Xception, ResNet50, InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Multiply, MaxPooling2D
from keras.models import Model


def build(size, seq_len , learning_rate ,
          optimizer_class ,\
          initial_weights ,\
          cnn_class ,\
          pre_weights , \
          lstm_conf , \
          cnn_train_type, classes = 1, dropout = 0.0):

    image_input = Input(shape=(seq_len, size, size, 3))
    pose_input = Input(shape=(seq_len, size, size, 3))

    if(cnn_train_type!='train'):
        if cnn_class.__name__ == "ResNet50":
            cnn = cnn_class(weights=pre_weights, include_top=False,input_shape =(size, size, 3))
        else:
            cnn = cnn_class(weights=pre_weights,include_top=False)
    else:
        cnn = cnn_class(include_top=False)
    
    #control Train_able of CNN
    if(cnn_train_type=='static'):
        for layer in cnn.layers:
            layer.trainable = False
    if(cnn_train_type=='retrain'):
        for layer in cnn.layers:
            layer.trainable = True

    pose_cnn = ResNet50(weights = None, include_top = False, input_shape = (size, size, 3))
    for layer in pose_cnn.layers:
      layer.trainable = True
    
    pose_cnn = Model(pose_cnn.input, pose_cnn.layers[-2].output)
    pose_cnn = TimeDistributed(pose_cnn)(pose_input)
    pose_cnn = Activation('sigmoid')(pose_cnn)

    cnn = TimeDistributed(cnn)(image_input)
    #print('cnn',cnn)
    #print('pose_cnn',pose_cnn)
    multiplied = Multiply()([cnn, pose_cnn])
    #print('multiplied',multiplied)
    cnn = TimeDistributed( MaxPooling2D(pool_size=(2,2)) ) (multiplied)
    #the resnet output shape is 1,1,20148 and need to be reshape for the ConvLSTM filters
    # if cnn_class.__name__ == "ResNet50":
        # cnn = Reshape((seq_len,4, 4, 128), input_shape=(seq_len,1, 1, 2048))(cnn)
    lstm = lstm_conf[0](**lstm_conf[1])(cnn)
    lstm = MaxPooling2D(pool_size=(2, 2))(lstm)   
    flat = Flatten()(lstm)

    flat = BatchNormalization()(flat)
    flat = Dropout(dropout)(flat)
    linear = Dense(1000)(flat)

    relu = Activation('relu')(linear)
    linear = Dense(256)(relu)
    linear = Dropout(dropout)(linear)
    relu = Activation('relu')(linear)
    linear = Dense(10)(relu)
    linear = Dropout(dropout)(linear)
    relu = Activation('relu')(linear)

    activation = 'sigmoid'
    loss_func = 'binary_crossentropy'

    if classes > 1:
        activation = 'softmax'
        loss_func = 'categorical_crossentropy'
    predictions = Dense(classes,  activation=activation)(relu)

    model = Model(inputs=[image_input, pose_input], outputs=predictions)
    optimizer = optimizer_class[0](lr=learning_rate, **optimizer_class[1])
    model.compile(optimizer=optimizer, loss=loss_func,metrics=['acc'])

    print(model.summary())

    return model