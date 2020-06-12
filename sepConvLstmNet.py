from tensorflow.keras import Input
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, ConvLSTM2D, Reshape, BatchNormalization, Activation, Conv2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D, Multiply, MaxPooling2D,Concatenate,Add
from tensorflow.keras.models import Model
from customLayers import SepConvLSTM2D

def getModel(size=224, seq_len=20 , cnn_weight=None, lstm_conf=None ):
    """parameters:
    size = height/width of each frame,
    seq_len = number of frames in each sequence,
    cnn_weight= None or 'imagenet'
       returns:
    model
    """
    image_input = Input(shape=(seq_len, size, size, 3))
    
    cnn = MobileNetV2(weights= cnn_weight, include_top=False,input_shape =(size, size, 3))

    for layer in cnn.layers:
        layer.trainable = True
 
    cnn = TimeDistributed(cnn)(image_input)

    lstm = SepConvLSTM2D(filters=256, kernel_size=(3, 3), padding='same', return_sequences=False)(cnn)
    lstm = MaxPooling2D(pool_size=(2, 2))(lstm)   
    flat = Flatten()(lstm)
    x = BatchNormalization()(flat)
    
    x = Dense(512,activation = 'relu')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(256,activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(10,activation='relu')(x)
    x = Dropout(0.1)(x)

    activation = 'sigmoid'
    
    if  activation == 'sigmoid':
        loss_func = 'binary_crossentropy'
        predictions = Dense(1, activation=activation)(x)
    elif  activation == 'softmax':
        loss_func = 'categorical_crossentropy'
        predictions = Dense(2,  activation=activation)(x)

    model = Model(inputs=[image_input], outputs=predictions)
    return model