from tensorflow.keras import Input
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, ConvLSTM2D, Reshape, BatchNormalization, Activation
from tensorflow.keras.layers.recurrent import LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers.wrappers import TimeDistributed
from tensorflow.keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
import sys
from tensorflow.keras.applications import Xception, ResNet50, InceptionV3, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Multiply, MaxPooling2D
from tensorflow.keras.models import Model


def getModel(size, seq_len , cnn_weight=None, lstm_conf ):
"""
parameter:
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
    pose_cnn = Activation('sigmoid')(pose_cnn)

    pose_cnn = TimeDistributed(pose_cnn)(pose_input)
    
    cnn = TimeDistributed(cnn)(image_input)

    multiplied = Multiply()([cnn, pose_cnn])
    cnn = TimeDistributed(MaxPooling2D(pool_size=(2,2))) (multiplied)
    
    lstm = lstm_conf[0](**lstm_conf[1])(cnn)
    lstm = MaxPooling2D(pool_size=(2, 2))(lstm)   
    flat = Flatten()(lstm)
    x = BatchNormalization()(flat)
    
    x = Dense(1000,activation = 'relu')(x)
    x = Dropout(dropout)(x)
    
    x = Dense(256,activation='relu')(x)
    x = Dropout(dropout)(x)

    x = Dense(10,activation='relu')(x)
    x = Dropout(dropout)(x)

    activation = 'sigmoid'
    loss_func = 'binary_crossentropy'

    if classes > 1:
        activation = 'softmax'
        loss_func = 'categorical_crossentropy'
    predictions = Dense(classes,  activation=activation)(x)

    model = Model(inputs=[image_input, pose_input], outputs=predictions)
    optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=optimizer, loss=loss_func,metrics=['acc'])

    print(model.summary())

    return model