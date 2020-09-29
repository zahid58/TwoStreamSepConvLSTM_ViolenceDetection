from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization, Activation, LeakyReLU, Add, Multiply
from tensorflow.keras.regularizers import l2

def getModel(size=224, seq_len=20 , cnn_weight=None, lstm_conf=None ):
    """parameters:
    size = height/width of each frame,
    seq_len = number of frames in each sequence,
    cnn_weight= None or 'imagenet'
       returns:
    model
    """
    inputs = Input(shape=(seq_len,size,size,3))

    #####################################################
    rgb = inputs
    rgb = Conv3D(
        16, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = Conv3D(
        16, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = MaxPooling3D(pool_size=(1,2,2))(rgb)

    rgb = Conv3D(
        16, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = Conv3D(
        16, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = MaxPooling3D(pool_size=(1,2,2))(rgb)

    rgb = Conv3D(
        32, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = Conv3D(
        32, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = MaxPooling3D(pool_size=(1,2,2))(rgb)

    rgb = Conv3D(
        32, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = Conv3D(
        32, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = MaxPooling3D(pool_size=(1,2,2))(rgb)

    #####################################################
    x = MaxPooling3D(pool_size=(8,1,1))(rgb)

    #####################################################
    x = Conv3D(
        64, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = Conv3D(
        64, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2,2,2))(x)

    x = Conv3D(
        64, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = Conv3D(
        64, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2,2,2))(x)

    x = Conv3D(
        128, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = Conv3D(
        128, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(1,3,3))(x)

    #####################################################
    x = Flatten()(x)
    x = Dense(128,activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.4)(x)
    pred = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=pred)
    return model


