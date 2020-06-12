from tensorflow.keras import Input
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, ConvLSTM2D, Reshape, BatchNormalization, Activation, Conv2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Lambda, Dense, GlobalAveragePooling2D, Multiply, MaxPooling2D, Concatenate, Add, AveragePooling2D 
from tensorflow.keras.initializers import glorot_uniform, he_normal
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K
from customLayers import SepConvLSTM2D
from customCnn import SeparableConvResnet

def identity_block(X, f, filters, stage, block):
    """parameters:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
        returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    X_shortcut = X

    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X



def convolutional_block(X, f, filters, stage, block, s = 2):
    """parameters:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
        returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    X_shortcut = X

    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1',
                        kernel_initializer = he_normal(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X



def tiny_resnet(input_shape=(224,224,3)):

    X_input = Input(input_shape)
    X = Conv2D(16, (7, 7), strides=(2, 2), name='conv1', kernel_initializer = he_normal(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, 5, filters=[16, 16, 32], stage=2, block='a', s=1)
    X = identity_block(X, 5, [16, 16, 32], stage=2, block='b')

    X = convolutional_block(X, 3, filters = [16, 16, 32], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [16, 16, 32], stage=3, block='b')
    X = identity_block(X, 3, [16, 16, 32], stage=3, block='c')

    X = convolutional_block(X, 3, filters = [32, 32, 64], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [32, 32, 64], stage=4, block='b')
    X = identity_block(X, 3, [32, 32, 64], stage=4, block='c')

    X = convolutional_block(X, 3, filters = [64, 64, 128], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [64, 64, 128], stage=5, block='b')
    return Model(inputs = X_input, outputs = X, name='tiny_resnet')



def getModel(size=224, seq_len=32 , cnn_weight=None):
    """parameters:
    size = height/width of each frame,
    seq_len = number of frames in each sequence,
    cnn_weight= None or 'imagenet'
       returns:
    model
    """
    image_input = Input(shape=(seq_len, size, size, 3),name='Input')
    
    #cnn = MobileNetV2(weights= cnn_weight, include_top=False,input_shape =(size, size, 3))
    cnn = tiny_resnet(input_shape=(size, size, 3))
    
    for layer in cnn.layers:
        layer.trainable = True
 
    cnn = TimeDistributed(cnn,name='CNN')(image_input)
    cnn = TimeDistributed(AveragePooling2D(pool_size=(2,2)),name='AveragePooling')(cnn)

    lstm = SepConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True,dropout=0.4, recurrent_dropout=0.4, name='SepConvLSTM2D')(cnn)

    # elementwise maxpooling / mean pooling
    TimeDistributedMean = Lambda(function=lambda x: K.mean(x, axis=1), output_shape=lambda shape: (shape[0],) + shape[2:] , name='TimeDistributedMean')
    lstm = TimeDistributedMean(lstm)

    x = Flatten()(lstm)
    #x = BatchNormalization()(x)

    dropout = 0.4
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropout)(x)

    activation = 'softmax'
    
    if  activation == 'sigmoid':
        predictions = Dense(1, activation=activation)(x)
    elif  activation == 'softmax':
        predictions = Dense(2,  activation=activation)(x)

    model = Model(inputs=[image_input], outputs=predictions)
    return model


