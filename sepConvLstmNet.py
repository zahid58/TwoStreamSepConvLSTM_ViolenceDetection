from tensorflow.keras import Input
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, ConvLSTM2D, Reshape, BatchNormalization, Activation, Conv2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import ELU, ReLU, LeakyReLU, Lambda, Dense, Bidirectional, Conv3D, GlobalAveragePooling2D, Multiply, MaxPooling3D, MaxPooling2D, Concatenate, Add, AveragePooling2D 
from tensorflow.keras.initializers import glorot_uniform, he_normal
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras import backend as K
from customLayers import SepConvLSTM2D

def getModel(size=224, seq_len=32 , cnn_weight = 'imagenet',cnn_trainable = True, weight_decay = 0.00005):
    """parameters:
    size = height/width of each frame,
    seq_len = number of frames in each sequence,
    cnn_weight= None or 'imagenet'
       returns:
    model
    """
    inputs = Input(shape=(seq_len, size, size, 6),name='input')
    
    frames_input = Lambda( lambda x: x[...,:3]  , name='frames_input', output_shape=(seq_len, size, size, 3) ) (inputs)
    frames_diff_input = Lambda( lambda x: x[...,3:] , name='frames_diff_input', output_shape=(seq_len, size, size, 3) ) (inputs)
    
    frames_cnn = MobileNetV2( input_shape = (size,size,3), alpha=0.35, weights='imagenet', include_top=False )
    frames_cnn = Model( inputs=[frames_cnn.layers[0].input],outputs=[frames_cnn.layers[-30].output] ) # taking only upto block 13

    print('> cnn_trainable : ', cnn_trainable)
    for layer in frames_cnn.layers:
        layer.trainable = cnn_trainable

    frames_diff_cnn = MobileNetV2( input_shape=(size,size,3), alpha=0.35, weights='imagenet', include_top = False )
    frames_diff_cnn = Model( inputs = [frames_diff_cnn.layers[0].input], outputs = [frames_diff_cnn.layers[-30].output] ) # taking only upto block 13
    
    print('> cnn_trainable : ', cnn_trainable)
    for layer in frames_diff_cnn.layers:
        layer.trainable = cnn_trainable

    frames_cnn = TimeDistributed( frames_cnn,name='frames_CNN' )( frames_input )
    frames_cnn = TimeDistributed( LeakyReLU(alpha=0.1), name='leaky_relu_1_' )( frames_cnn)
    frames_cnn = TimeDistributed( Dropout(0.2) ,name='dropout_1_' )(frames_cnn)
  
    frames_diff_cnn = TimeDistributed( frames_diff_cnn,name='frames_diff_CNN' )(frames_diff_input)
    frames_diff_cnn = TimeDistributed( LeakyReLU(alpha=0.1), name='leaky_relu_2_' )(frames_diff_cnn)
    frames_diff_cnn = TimeDistributed( Dropout(0.2) ,name='dropout_2_' )(frames_diff_cnn)

    cnn = Concatenate(axis=-1, name='concatenate_')([frames_cnn, frames_diff_cnn])

    lstm = SepConvLSTM2D( filters = 128, kernel_size=(3, 3), padding='same', return_sequences=True, dropout=0.4, recurrent_dropout=0.4, name='SepConvLSTM2D_1', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(cnn)
    lstm = BatchNormalization( axis = 4 )(lstm)

    x = Conv3D(
        64, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same',kernel_regularizer=l2(weight_decay),name='3d_conv_1')(lstm)
   
    x = MaxPooling3D(pool_size=(8,1,1))(x)
    
    x = Conv3D(
        64, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same',kernel_regularizer=l2(weight_decay),name='3d_conv_2')(x)
  
    x = MaxPooling3D(pool_size=(2,2,2))(x)

    x = Conv3D(
        64, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same',kernel_regularizer=l2(weight_decay), name="3d_conv_3")(x)
    
    x = MaxPooling3D(pool_size=(1,3,3))(x)

    x = Flatten()(x)
    dropout = 0.4
    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(dropout)(x)
    x = Dense(16)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(dropout)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[inputs], outputs=predictions)
    return model




