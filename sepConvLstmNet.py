from tensorflow.keras import Input
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, ConvLSTM2D, Reshape, BatchNormalization, Activation, Conv2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Lambda, Dense, GlobalAveragePooling2D, Multiply, MaxPooling2D, Concatenate, Add, AveragePooling2D 
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K
from customLayers import SepConvLSTM2D
from customCnn import SeparableConvResnet

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
    cnn = TimeDistributed(AveragePooling2D(pool_size=(2,2)))(cnn)

    lstm = SepConvLSTM2D(filters=512, kernel_size=(3, 3), padding='same', return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(cnn)
    lstm = SepConvLSTM2D(filters=512, kernel_size=(3, 3), padding='same', return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(lstm)

    # elementwise maxpooling / mean pooling
    TimeDistributedMean = Lambda(function=lambda x: K.mean(x, axis=1), output_shape=lambda shape: (shape[0],) + shape[2:])
    lstm = TimeDistributedMean(lstm)

    lstm = Flatten()(lstm)
    x = BatchNormalization()(lstm)

    dropout = 0.4
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(dropout)(x)

    activation = 'sigmoid'
    
    if  activation == 'sigmoid':
        predictions = Dense(1, activation=activation)(x)
    elif  activation == 'softmax':
        predictions = Dense(2,  activation=activation)(x)

    model = Model(inputs=[image_input], outputs=predictions)
    return model


