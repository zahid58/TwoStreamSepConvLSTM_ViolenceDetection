from tensorflow.keras import Input
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, ConvLSTM2D, Reshape, BatchNormalization, Activation, Conv2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import ELU, ReLU, LeakyReLU, Lambda, Dense, Bidirectional, Conv3D, GlobalAveragePooling2D, Multiply, MaxPooling3D, MaxPooling2D, Concatenate, Add, AveragePooling2D 
from tensorflow.keras.initializers import glorot_uniform, he_normal
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50V2, VGG16
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras import backend as K

# the original model is implemented in pytorch in the link below.
# https://github.com/swathikirans/violence-recognition-pytorch/blob/master/createModel.py
# we had to use ResNet50 in place of AlexNet because of limitation of gpu memory. It achieves the same accuracy on hockey and movies. So, this change does not hurt performance of the model

def getConvLSTM(size=224, seq_len=32 , cnn_weight = 'imagenet',cnn_trainable = True, weight_decay = 0.00005, frame_diff_interval = 1):
    
    frames_input = Input(shape=(seq_len, size, size, 3),name='frames_input')
    
    frames_cnn = ResNet50V2(weights = cnn_weight, include_top = False,input_shape = (size, size, 3))

    print('> cnn_trainable : ', cnn_trainable)
    for layer in frames_cnn.layers:
        layer.trainable = cnn_trainable

    frames_cnn = TimeDistributed( frames_cnn,name='frames_CNN' )( frames_input )

    lstm_dropout = 0.2
    lstm = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding='same', return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='ConvLSTM2D_', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(frames)
    lstm = MaxPooling2D((2,2))(lstm)
    x = Flatten()(lstm) 

    dense_dropout = 0.3
    # x = Dense(1000)(x)   # one FC layer reduced to fit in GPU and reduce redundant parameters
    # x = ReLU()(x)
    x = Dense(256)(x)
    x = ReLU()(x)
    x = Dropout(dense_dropout)(x)
    x = Dense(10)(x)
    x = ReLU()(x)
    x = Dropout(dense_dropout)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=frames_input, outputs=predictions)
    return model



# the following model is built following the paper given below
# https://openaccess.thecvf.com/content_ECCVW_2018/papers/11130/Hanson_Bidirectional_Convolutional_LSTM_for_the_Detection_of_Violence_in_Videos_ECCVW_2018_paper.pdf

def getBiConvLSTM(size=224, seq_len=32 , cnn_weight = 'imagenet',cnn_trainable = True, weight_decay = 0.00005, frame_diff_interval = 1):
    
    frames_input = Input(shape=(seq_len, size, size, 3),name='frames_input')
    
    frames_cnn = VGG16(weights = cnn_weight, include_top = False, input_shape = (size, size, 3))

    print('> cnn_trainable : ', cnn_trainable)
    for layer in frames_cnn.layers:
        layer.trainable = cnn_trainable

    frames_cnn = TimeDistributed( frames_cnn,name='frames_CNN' )( frames_input )

    lstm_dropout = 0.2
    lstm = Bidirectional(ConvLSTM2D(filters=256, kernel_size=(3, 3), padding='same', return_sequences=True, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='ConvLSTM2D_', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay)))(frames_cnn)
    
    elementwise_maxpooling = Lambda(function=lambda x: K.mean(x, axis=1), output_shape=lambda shape: (shape[0],) + shape[2:] , name='ElementWiseMaxPooling')
    lstm = elementwise_maxpooling(lstm)

    x = Flatten()(lstm) 

    dense_dropout = 0.3
    # x = Dense(1000)(x)   # one FC layer reduced to fit in GPU and reduce redundant parameters
    # x = ReLU()(x)
    x = Dense(256)(x)
    x = ReLU()(x)
    x = Dropout(dense_dropout)(x)
    x = Dense(10)(x)
    x = ReLU()(x)
    x = Dropout(dense_dropout)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=frames_input, outputs=predictions)
    return model