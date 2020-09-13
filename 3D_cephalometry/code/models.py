import tensorflow as tf
import keras
from keras import backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *


def load_3d_unet(pretrained_weights=None, input_shape=(136, 128, 144, 1), num_labels=4, init_filter=24):
    inputs = Input(shape=input_shape)
    conv1 = ZeroPadding3D((1, 1, 1))(inputs)
    conv1 = Conv3D(init_filter, (3, 3, 3), strides=(1, 1, 1))(conv1)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = ZeroPadding3D((1, 1, 1))(conv1)
    conv1 = Conv3D(init_filter, (3, 3, 3), strides=(1, 1, 1))(conv1)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = ZeroPadding3D((1, 1, 1))(pool1)
    conv2 = Conv3D(init_filter * 2, (3, 3, 3), strides=(1, 1, 1))(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = ZeroPadding3D((1, 1, 1))(conv2)
    conv2 = Conv3D(init_filter * 2, (3, 3, 3), strides=(1, 1, 1))(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = ZeroPadding3D((1, 1, 1))(pool2)
    conv3 = Conv3D(init_filter * 4, (3, 3, 3), strides=(1, 1, 1))(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = ZeroPadding3D((1, 1, 1))(conv3)
    conv3 = Conv3D(init_filter * 2, (3, 3, 3), strides=(1, 1, 1))(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Activation('relu')(conv3)

    up1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv3), conv2], axis=-1)
    conv4 = ZeroPadding3D((1, 1, 1))(up1)
    conv4 = Conv3D(init_filter * 4, (3, 3, 3), strides=(1, 1, 1))(conv4)
    conv4 = BatchNormalization(axis=-1)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = ZeroPadding3D((1, 1, 1))(conv4)
    conv4 = Conv3D(init_filter * 2, (3, 3, 3), strides=(1, 1, 1))(conv4)
    conv4 = BatchNormalization(axis=-1)(conv4)
    conv4 = Activation('relu')(conv4)

    up2 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1], axis=-1)
    conv5 = ZeroPadding3D((1, 1, 1))(up2)
    conv5 = Conv3D(init_filter * 2, (3, 3, 3), strides=(1, 1, 1))(conv5)
    conv5 = BatchNormalization(axis=-1)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = ZeroPadding3D((1, 1, 1))(conv5)

    # segment
    # conv5s = Conv3D(init_filter*2, (3, 3, 3), strides=(1, 1, 1))(conv5)
    # conv5s = BatchNormalization(axis = -1)(conv5s)
    # conv5s = Activation('relu')(conv5s)
    # outputs1=Conv3D(2, (1, 1, 1), activation='sigmoid')(conv5s)

    # landmark
    conv5l = Conv3D(init_filter * 2, (3, 3, 3), strides=(1, 1, 1))(conv5)
    conv5l = BatchNormalization(axis=-1)(conv5l)
    conv5l = Activation('relu')(conv5l)
    dropout = Dropout(0.4)(conv5l)
    outputs2 = Conv3D(num_labels, (1, 1, 1), activation='sigmoid')(dropout)

    model = Model(inputs=inputs, outputs=outputs2)
    model.compile(optimizer=Adam(lr=1e-4), loss=['mean_squared_error'], metrics=['accuracy'])
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model


def load_3d_res_unet(pretrained_weights=None, input_shape=(136, 128, 144, 1), num_labels=4, init_filter=24):
    inputs = Input(shape=input_shape)
    conv1 = ZeroPadding3D((1, 1, 1))(inputs)
    conv1c = Conv3D(init_filter, (3, 3, 3), strides=(2, 2, 2))(conv1)
    conv1c = BatchNormalization(axis=-1)(conv1c)
    conv1c = Activation('relu')(conv1c)
    conv1 = Conv3D(init_filter, (3, 3, 3), strides=(1, 1, 1))(conv1)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = ZeroPadding3D((1, 1, 1))(conv1)
    conv1 = Conv3D(init_filter, (3, 3, 3), strides=(1, 1, 1))(conv1)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    merge1 = add([pool1, conv1c])

    conv2 = ZeroPadding3D((1, 1, 1))(merge1)
    conv2c = Conv3D(init_filter * 2, (3, 3, 3), strides=(2, 2, 2))(conv2)
    conv2c = BatchNormalization(axis=-1)(conv2c)
    conv2c = Activation('relu')(conv2c)
    conv2 = Conv3D(init_filter * 2, (3, 3, 3), strides=(1, 1, 1))(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = ZeroPadding3D((1, 1, 1))(conv2)
    conv2 = Conv3D(init_filter * 2, (3, 3, 3), strides=(1, 1, 1))(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    merge2 = add([pool2, conv2c])

    conv3 = ZeroPadding3D((1, 1, 1))(merge2)
    conv3c = Conv3D(init_filter * 2, (3, 3, 3), strides=(1, 1, 1))(conv3)
    conv3c = BatchNormalization(axis=-1)(conv3c)
    conv3c = Activation('relu')(conv3c)
    conv3 = Conv3D(init_filter * 4, (3, 3, 3), strides=(1, 1, 1))(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = ZeroPadding3D((1, 1, 1))(conv3)
    conv3 = Conv3D(init_filter * 2, (3, 3, 3), strides=(1, 1, 1))(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Activation('relu')(conv3)
    merge3 = add([conv3, conv3c])

    up1 = concatenate([UpSampling3D(size=(2, 2, 2))(merge3), conv2], axis=-1)
    conv4 = ZeroPadding3D((1, 1, 1))(up1)
    conv4c = Conv3D(init_filter * 2, (3, 3, 3), strides=(1, 1, 1))(conv4)
    conv4c = BatchNormalization(axis=-1)(conv4c)
    conv4c = Activation('relu')(conv4c)
    conv4 = Conv3D(init_filter * 4, (3, 3, 3), strides=(1, 1, 1))(conv4)
    conv4 = BatchNormalization(axis=-1)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = ZeroPadding3D((1, 1, 1))(conv4)
    conv4 = Conv3D(init_filter * 2, (3, 3, 3), strides=(1, 1, 1))(conv4)
    conv4 = BatchNormalization(axis=-1)(conv4)
    conv4 = Activation('relu')(conv4)
    merge4 = add([conv4, conv4c])

    up2 = concatenate([UpSampling3D(size=(2, 2, 2))(merge4), conv1], axis=-1)
    conv5 = ZeroPadding3D((1, 1, 1))(up2)
    conv5 = Conv3D(init_filter * 2, (3, 3, 3), strides=(1, 1, 1))(conv5)
    conv5 = BatchNormalization(axis=-1)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = ZeroPadding3D((1, 1, 1))(conv5)

    # conv5s = Conv3D(init_filter*2, (3, 3, 3), strides=(1, 1, 1))(conv5)
    # conv5s = BatchNormalization(axis = -1)(conv5s)
    # conv5s = Activation('relu')(conv5s)
    # outputs1=Conv3D(4, (1, 1, 1), activation='sigmoid')(conv5s)

    conv5l = Conv3D(init_filter * 2, (3, 3, 3), strides=(1, 1, 1))(conv5)
    conv5l = BatchNormalization(axis=-1)(conv5l)
    conv5l = Activation('relu')(conv5l)
    dropout = Dropout(0.4)(conv5l)
    outputs2 = Conv3D(num_labels, (1, 1, 1), activation='sigmoid')(dropout)

    model = Model(inputs=inputs, outputs=[outputs2])
    model.compile(optimizer=Adam(lr=1e-3), loss=['mean_squared_error'], metrics=['accuracy'])
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model


def load_3d_dense_unet(pretrained_weights=None, input_shape=(136, 128, 144, 1), num_labels=4, init_filter=24):
    inputs = Input(shape=input_shape)
    conv11 = Conv3D(init_filter, (3, 3, 3), activation='relu', padding='same')(inputs)
    conc11 = concatenate([inputs, conv11], axis=4)
    conv12 = Conv3D(init_filter, (3, 3, 3), activation='relu', padding='same')(conc11)
    conc12 = concatenate([inputs, conv12], axis=4)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conc12)

    conv21 = Conv3D(init_filter * 2, (3, 3, 3), activation='relu', padding='same')(pool1)
    conc21 = concatenate([pool1, conv21], axis=4)
    conv22 = Conv3D(init_filter * 2, (3, 3, 3), activation='relu', padding='same')(conc21)
    conc22 = concatenate([pool1, conv22], axis=4)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conc22)

    conv31 = Conv3D(init_filter * 4, (3, 3, 3), activation='relu', padding='same')(pool2)
    conc31 = concatenate([pool2, conv31], axis=4)
    conv32 = Conv3D(init_filter * 4, (3, 3, 3), activation='relu', padding='same')(conc31)
    conc32 = concatenate([pool2, conv32], axis=4)

    up1 = concatenate([Conv3DTranspose(init_filter * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc32), conc22],
                      axis=4)
    conv41 = Conv3D(init_filter * 2, (3, 3, 3), activation='relu', padding='same')(up1)
    conc41 = concatenate([up1, conv41], axis=4)
    conv42 = Conv3D(init_filter * 2, (3, 3, 3), activation='relu', padding='same')(conc41)
    conc42 = concatenate([up1, conv42], axis=4)

    up2 = concatenate([Conv3DTranspose(init_filter, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc42), conc12],
                      axis=4)
    conv51 = Conv3D(init_filter, (3, 3, 3), activation='relu', padding='same')(up2)
    conc51 = concatenate([up2, conv51], axis=4)
    conv52 = Conv3D(init_filter, (3, 3, 3), activation='relu', padding='same')(conc51)
    conc52 = concatenate([up2, conv52], axis=4)

    #     conv5s = Conv3D(init_filter*2, (3, 3, 3),activation='relu',padding='same')(conc52)
    #     dropouts = Dropout(0.3)(conv5s)
    #     outputs1=Conv3D(4, (1, 1, 1), activation='sigmoid')(dropouts)

    conv5l = Conv3D(init_filter * 2, (3, 3, 3), activation='relu', padding='same')(conc52)
    dropout = Dropout(0.3)(conv5l)
    outputs2 = Conv3D(num_labels, (1, 1, 1), activation='sigmoid')(dropout)

    model = Model(inputs=inputs, outputs=outputs2)
    model.compile(optimizer=Adam(lr=1e-5), loss=['mean_squared_error'], metrics=['accuracy'])
    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model











