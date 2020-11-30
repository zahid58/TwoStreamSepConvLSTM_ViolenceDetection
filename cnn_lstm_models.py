
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, ConvLSTM2D, Reshape, BatchNormalization, Activation, Conv2D, LayerNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import TimeDistributed, RepeatVector,Permute, Multiply
from tensorflow.keras.applications import MobileNetV2, VGG16
from tensorflow.keras.layers import ELU, ReLU, LeakyReLU, Lambda, Dense, Bidirectional, Conv3D, GlobalAveragePooling2D, Multiply, MaxPooling3D, MaxPooling2D, Concatenate, Add, AveragePooling2D 
from tensorflow.keras.initializers import glorot_uniform, he_normal
from tensorflow.keras.models import Model
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras import backend as K
from sep_conv_rnn import SepConvLSTM2D
import torchvision.models as models
from torchsummary import summary
import torch.nn as tnn
import torch
from torch.autograd import Variable
import numpy as np
from pytorch2keras import pytorch_to_keras
from yolo_models import get_yolo3_model

# loads alexnet pretrained on imagenet and converts it to a keras model 

def getAlexNet(pretrained = True):
    print("> loading alexnet pytorch model...")
    K.set_image_data_format("channels_first")
    alex = models.alexnet(pretrained = pretrained, progress = True)
    alex = tnn.Sequential(*list(alex.children())[:-2])
    input_np = np.random.uniform(0, 1, (1, 3,224,224))
    input_var = Variable(torch.FloatTensor(input_np))
    print("> converting alexnet to keras model...")
    model = pytorch_to_keras(alex, input_var, [(3, 224, 224)], change_ordering = True, verbose = False)
    alex = None
    K.set_image_data_format("channels_last")
    print("> conversion done...")
    return model

# loads vgg13 pretrained on imagenet and converts it to a keras model 

def getVGG13(pretrained = True):
    print("> loading VGG13 pytorch model...")
    K.set_image_data_format("channels_first")
    vgg13 = models.vgg13(pretrained = pretrained, progress = True)
    vgg13.features = tnn.Sequential(*list(vgg13.features.children())[:-1])
    vgg13 = tnn.Sequential(*list(vgg13.children())[:-2])
    input_np = np.random.uniform(0, 1, (1, 3,224,224))
    input_var = Variable(torch.FloatTensor(input_np))
    print("> converting VGG13 to keras model...")
    model = pytorch_to_keras(vgg13, input_var, [(3, 224, 224)], change_ordering = True, verbose = False)
    vgg13 = None
    K.set_image_data_format("channels_last")
    print("> conversion done...")
    return model



# this model is built following the paper given below
# https://arxiv.org/abs/1709.06531    
# https://github.com/swathikirans/violence-recognition-pytorch/blob/master/createModel.py

def getConvLSTM(size=224, seq_len=32 , cnn_weight = 'imagenet', cnn_trainable = True, weight_decay = 1e-5, frame_diff_interval = 1, seed = 42, mode = "only_differences"):
    """parameters:
    size = height/width of each frame,
    seq_len = number of frames in each sequence,
    cnn_weight= None or 'imagenet'
    mode = "only_frames" or "only_differences"
       returns:
    model
    """    
    if mode == "only_frames":
        frames_input = Input(shape=(seq_len, size, size, 3),name='frames_input')
    elif mode == "only_differences":
        frames_input = Input(shape=(seq_len-frame_diff_interval, size, size, 3),name='frames_input')

    frames_cnn = getAlexNet()

    print('> cnn_trainable : ', cnn_trainable)
    for layer in frames_cnn.layers:
        layer.trainable = cnn_trainable

    frames_cnn = TimeDistributed( frames_cnn,name='frames_CNN' )( frames_input )

    lstm = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding='same', name='ConvLSTM2D_', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(frames_cnn)
    lstm = MaxPooling2D((2,2))(lstm)
    x = Flatten()(lstm) 

    x = Dense(1000, activation='relu')(x) 
    x = BatchNormalization(axis=-1)(x) 
    x = Dense(256,activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=frames_input, outputs=predictions)
    return model



# this model is built following the paper given below
# https://openaccess.thecvf.com/content_ECCVW_2018/papers/11130/Hanson_Bidirectional_Convolutional_LSTM_for_the_Detection_of_Violence_in_Videos_ECCVW_2018_paper.pdf

def getBiConvLSTM(size=224, seq_len=32 , cnn_weight = 'imagenet',cnn_trainable = True, weight_decay = 1e-5, frame_diff_interval = 1, seed = 42, mode = "only_differences"):
    """parameters:
    size = height/width of each frame,
    seq_len = number of frames in each sequence,
    cnn_weight= None or 'imagenet'
    mode = "only_frames" or "only_differences"
       returns:
    model
    """    
    if mode == "only_frames":
        frames_input = Input(shape=(seq_len, size, size, 3),name='frames_input')
    elif mode == "only_differences":
        frames_input = Input(shape=(seq_len-frame_diff_interval, size, size, 3),name='frames_input')
    
    frames_cnn = getVGG13()

    print('> cnn_trainable : ', cnn_trainable)
    for layer in frames_cnn.layers:
        layer.trainable = cnn_trainable

    frames_cnn = TimeDistributed( frames_cnn,name='frames_CNN' )( frames_input )

    lstm = Bidirectional(ConvLSTM2D(filters=256, kernel_size=(3, 3), padding='same', return_sequences=True, name='ConvLSTM2D_', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay)))(frames_cnn)
    
    elementwise_maxpooling = Lambda(function=lambda x: K.mean(x, axis=1), output_shape=lambda shape: (shape[0],) + shape[2:] , name='ElementWiseMaxPooling')
    lstm = elementwise_maxpooling(lstm)
    
    lstm = MaxPooling2D((2,2))(lstm)
    x = Flatten()(lstm) 

    x = Dense(1000, activation='tanh')(x)
    x = Dense(256, activation ='tanh')(x)
    x = Dense(10, activation='tanh')(x)
    predictions = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=frames_input, outputs=predictions)
    return model



# This is a model with two streams, frames and frame_differences

def BiStreamCNNLSTM(size=224, seq_len=32 , cnn_weight = 'imagenet',cnn_trainable = True, lstm_type = 'sepconv',weight_decay = 1e-5, frame_diff_interval = 1, mode = "both", cnn_dropout = 0.2, lstm_dropout = 0.2, dense_dropout = 0.3, seed = 42):
    """parameters:
    size = height/width of each frame,
    seq_len = number of frames in each sequence,
    cnn_weight= None or 'imagenet'
    mode = "only_frames" or "only_differences" or "both"
       returns:
    model
    """
    print('cnn dropout : ', cnn_dropout)
    print('dense dropout : ', dense_dropout)
    print('lstm dropout :', lstm_dropout)

    if mode == "both":
        frames = True
        differences = True
    elif mode == "only_frames":
        frames = True
        differences = False
    elif mode == "only_differences":
        frames = False
        differences = True

    if frames:

        frames_input = Input(shape=(seq_len, size, size, 3),name='frames_input')
        frames_cnn = MobileNetV2( input_shape = (size,size,3), alpha=0.35, weights='imagenet', include_top = False)
        frames_cnn = Model( inputs=[frames_cnn.layers[0].input],outputs=[frames_cnn.layers[-30].output] ) # taking only upto block 13
        
        for layer in frames_cnn.layers:
            layer.trainable = cnn_trainable

        frames_cnn = TimeDistributed( frames_cnn,name='frames_CNN' )( frames_input )
        frames_cnn = TimeDistributed( LeakyReLU(alpha=0.1), name='leaky_relu_1_' )( frames_cnn)
        frames_cnn = TimeDistributed( Dropout(cnn_dropout, seed=seed) ,name='dropout_1_' )(frames_cnn)
        if lstm_type == 'sepconv':
            frames_lstm = SepConvLSTM2D( filters = 64, kernel_size=(3, 3), padding='same', return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='SepConvLSTM2D_1', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(frames_cnn)
        elif lstm_type == 'conv':
            frames_lstm = ConvLSTM2D( filters = 64, kernel_size=(3, 3), padding='same', return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='ConvLSTM2D_1', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(frames_cnn)
        else:
            raise Exception("LSTM type not recognized!")   
        frames_lstm = BatchNormalization( axis = -1 )(frames_lstm)
        
    if differences:

        frames_diff_input = Input(shape=(seq_len - frame_diff_interval, size, size, 3),name='frames_diff_input')
        frames_diff_cnn = MobileNetV2( input_shape=(size,size,3), alpha=0.35, weights='imagenet', include_top = False)
        frames_diff_cnn = Model( inputs = [frames_diff_cnn.layers[0].input], outputs = [frames_diff_cnn.layers[-30].output] ) # taking only upto block 13
    
        for layer in frames_diff_cnn.layers:
            layer.trainable = cnn_trainable
    
        frames_diff_cnn = TimeDistributed( frames_diff_cnn,name='frames_diff_CNN' )(frames_diff_input)
        frames_diff_cnn = TimeDistributed( LeakyReLU(alpha=0.1), name='leaky_relu_2_' )(frames_diff_cnn)
        frames_diff_cnn = TimeDistributed( Dropout(cnn_dropout, seed=seed) ,name='dropout_2_' )(frames_diff_cnn)
        if lstm_type == 'sepconv':
            frames_diff_lstm = SepConvLSTM2D( filters = 64, kernel_size=(3, 3), padding='same', return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='SepConvLSTM2D_2', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(frames_diff_cnn)
        elif lstm_type == 'conv':    
            frames_diff_lstm = ConvLSTM2D( filters = 64, kernel_size=(3, 3), padding='same', return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='ConvLSTM2D_2', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(frames_diff_cnn)
        else:
            raise Exception("LSTM type not recognized!")
        frames_diff_lstm = BatchNormalization( axis = -1 )(frames_diff_lstm)


    if mode == "both":
        lstm = Concatenate(axis=-1, name='concatenate_')([frames_lstm, frames_diff_lstm])
    elif mode == "only_frames":
        lstm = frames_lstm
    elif mode == "only_differences":
        lstm = frames_diff_lstm

    lstm = MaxPooling2D((2,2) , name = 'max_pooling_')(lstm)
    
    x = Flatten()(lstm) 
  
    
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(dense_dropout, seed = seed)(x)
    x = Dense(16)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(dense_dropout, seed = seed)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    if mode == "both":
        model = Model(inputs=[frames_input, frames_diff_input], outputs=predictions)
    elif mode == "only_frames":
        model = Model(inputs=frames_input, outputs=predictions)
    elif mode == "only_differences":
        model = Model(inputs=frames_diff_input, outputs=predictions)

    return model



# This is the proposed model for violent activity detection

def getProposedModel(size=224, seq_len=32 , cnn_weight = 'imagenet',cnn_trainable = True, yolo_trainable = True, lstm_type = 'sepconv',weight_decay = 1e-5, frame_diff_interval = 1, mode = "all", cnn_dropout = 0.2, lstm_dropout = 0.2, dense_dropout = 0.3, seed = 42):
    """parameters:
    size = height/width of each frame,
    seq_len = number of frames in each sequence,
    cnn_weight= None or 'imagenet'
    mode = "only_frames" or "only_differences" or "only_yolo" or "all"
       returns:
    model
    """
    print('cnn_trainable:',cnn_trainable)
    print('-----------------------------')
    print('cnn dropout : ', cnn_dropout)
    print('dense dropout : ', dense_dropout)
    print('lstm dropout :', lstm_dropout)

    frames = True
    differences = True

    if frames:

        frames_input = Input(shape=(seq_len, size, size, 3),name='frames_input')
        frames_cnn = MobileNetV2( input_shape=(size,size,3), alpha=0.35, weights='imagenet', include_top = False)
        frames_cnn = Model( inputs = [frames_cnn.layers[0].input], outputs = [frames_cnn.layers[-30].output] ) # taking only upto block 13
        
        frames_cnn.trainable = cnn_trainable

        frames_cnn = TimeDistributed( frames_cnn,name='frames_CNN' )(frames_input, training = False)
        frames_cnn = TimeDistributed( LeakyReLU(alpha=0.1), name='leaky_relu_1_' )(frames_cnn)
        frames_cnn = TimeDistributed( Dropout(cnn_dropout, seed=seed) ,name='dropout_1_' )(frames_cnn)
        frames_cnn = LayerNormalization( axis = -1 )(frames_cnn)

        if lstm_type == 'sepconv':
            frames_lstm = SepConvLSTM2D( filters = 64, kernel_size=(3, 3), padding='same', return_sequences=True, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='SepConvLSTM2D_1', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(frames_cnn)
        elif lstm_type == 'conv':    
            frames_lstm = ConvLSTM2D( filters = 64, kernel_size=(3, 3), padding='same', return_sequences=True, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='ConvLSTM2D_1', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(frames_cnn)
        else:
            raise Exception("LSTM type not recognized!")
        frames_lstm = LayerNormalization( axis = -1 )(frames_lstm)

        frames_cnn_flattened = TimeDistributed(Flatten())(frames_cnn)
        frames_lstm_flattened  = TimeDistributed(Flatten())(frames_lstm)
        frames_flattened = Concatenate(axis=-1)([frames_cnn_flattened, frames_lstm_flattened])
        # print("shape0:", frames_lstm_flattened.shape)
        frames_attention = Dense(8, activation='tanh')(frames_flattened)
        frames_attention = Dense(1, activation='tanh')(frames_attention)
        # print("shape1:", frames_attention.shape)
        frames_attention = Flatten()(frames_attention)
        # print("shape2:", frames_attention.shape)
        frames_attention = Activation('softmax')(frames_attention)
        # print("shape3:", frames_attention.shape)
        frames_attention = RepeatVector(7*7*64)(frames_attention)
        # print("shape4:", frames_attention.shape)
        frames_attention = Permute([2, 1])(frames_attention)
        # print("shape5:", frames_attention.shape)
        frames_attention = TimeDistributed(Reshape( (7,7,64) ))(frames_attention)
        # print("shape6:", frames_attention.shape)
        frames_lstm = Multiply()([frames_lstm, frames_attention])
        frames_lstm = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(None,7,7,64))(frames_lstm)

    if differences:

        frames_diff_input = Input(shape=(seq_len - frame_diff_interval, size, size, 3),name='frames_diff_input')
        frames_diff_cnn = MobileNetV2( input_shape=(size,size,3), alpha=0.35, weights='imagenet', include_top = False)
        frames_diff_cnn = Model( inputs = [frames_diff_cnn.layers[0].input], outputs = [frames_diff_cnn.layers[-30].output] ) # taking only upto block 13
        
        frames_diff_cnn.trainable = cnn_trainable
        
        frames_diff_cnn = TimeDistributed( frames_diff_cnn,name='frames_diff_CNN' )(frames_diff_input, training = False)
        frames_diff_cnn = TimeDistributed( LeakyReLU(alpha=0.1), name='leaky_relu_2_' )(frames_diff_cnn)
        frames_diff_cnn = TimeDistributed( Dropout(cnn_dropout, seed=seed) ,name='dropout_2_' )(frames_diff_cnn)
        frames_diff_cnn = LayerNormalization( axis = -1 )(frames_diff_cnn)

        if lstm_type == 'sepconv':
            frames_diff_lstm = SepConvLSTM2D( filters = 64, kernel_size=(3, 3), padding='same', return_sequences=True, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='SepConvLSTM2D_2', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(frames_diff_cnn)
        elif lstm_type == 'conv':    
            frames_diff_lstm = ConvLSTM2D( filters = 64, kernel_size=(3, 3), padding='same', return_sequences=True, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='ConvLSTM2D_2', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(frames_diff_cnn)
        else:
            raise Exception("LSTM type not recognized!")
        frames_diff_lstm = LayerNormalization( axis = -1 )(frames_diff_lstm)

        frames_diff_cnn_flattened = TimeDistributed(Flatten())(frames_diff_cnn)
        frames_diff_lstm_flattened  = TimeDistributed(Flatten())(frames_diff_lstm)
        frames_diff_flattened = Concatenate(axis=-1)([frames_diff_cnn_flattened, frames_diff_lstm_flattened])
        # print("shape0:", frames_lstm_flattened.shape)
        frames_diff_attention = Dense(8, activation='tanh')(frames_diff_flattened)
        frames_diff_attention = Dense(1, activation='tanh')(frames_diff_attention)
        # print("shape1:", frames_diff_attention.shape)
        frames_diff_attention = Flatten()(frames_diff_attention)
        # print("shape2:", frames_diff_attention.shape)
        frames_diff_attention = Activation('softmax')(frames_diff_attention)
        # print("shape3:", frames_diff_attention.shape)
        frames_diff_attention = RepeatVector(7*7*64)(frames_diff_attention)
        # print("shape4:", frames_diff_attention.shape)
        frames_diff_attention = Permute([2, 1])(frames_diff_attention)
        # print("shape5:", frames_diff_attention.shape)
        frames_diff_attention = TimeDistributed(Reshape( (7,7,64) ))(frames_diff_attention)
        # print("shape6:", frames_diff_attention.shape)
        frames_diff_lstm = Multiply()([frames_diff_lstm, frames_diff_attention])
        frames_diff_lstm = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(7,7,64))(frames_diff_lstm)

    frames_lstm = expand_dims(frames_lstm, axis=1)
    # print("shape_a:",frames_lstm.shape)
    frames_diff_lstm = expand_dims(frames_diff_lstm, axis=1)
    # print("shape_b:",frames_diff_lstm.shape)
    lstm = Concatenate(axis=1)([frames_lstm , frames_diff_lstm])

    lstm_flattened  = TimeDistributed(Flatten())(lstm)
    # print("shape0:", lstm_flattened.shape)
    attention = Dense(8, activation='tanh')(lstm_flattened)
    attention = Dense(1, activation='tanh')(attention)
    # print("shape1:", attention.shape)
    attention = Flatten()(attention)
    # print("shape2:", attention.shape)
    attention = Activation('softmax')(attention)
    # print("shape3:", attention.shape)
    attention = RepeatVector(7*7*64)(attention)
    # print("shape4:", attention.shape)
    attention = Permute([2, 1])(attention)
    # print("shape5:", attention.shape)
    attention = TimeDistributed(Reshape( (7,7,64) ))(attention)
    # print("shape6:", attention.shape)
    lstm = Multiply()([lstm, attention])
    lstm = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(7,7,64))(lstm)
    lstm = LayerNormalization(axis=-1)(lstm)
    
    lstm = MaxPooling2D((2,2))(lstm)
    x = Flatten()(lstm) 
  
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(dense_dropout, seed = seed)(x)
    x = Dense(16)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(dense_dropout, seed = seed)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    if mode == "all":
        model = Model(inputs=[frames_input, frames_diff_input], outputs=predictions)
    elif mode == "only_frames":
        model = Model(inputs=frames_input, outputs=predictions)
    elif mode == "only_differences":
        model = Model(inputs=frames_diff_input, outputs=predictions)

    return model