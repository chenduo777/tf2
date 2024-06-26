import tensorflow as tf
from tensorflow import keras

def Generate(input_shape=(1, 1, 128), name='Generate'):
    inputs = keras.Input(shape=input_shape)

    x = keras.layers.Conv2DTranspose(512, 4, strides=1, padding='valid', use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', use_bias=False)(x)
    output = keras.layers.Activation('tanh')(x)
    return keras.Model(inputs=inputs, outputs=output, name=name)

def Discriminate(input_shape=(64, 64, 3), name='Discriminate'):
    inputs = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(64, 4, strides=2, padding='same', use_bias=False)(inputs)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv2D(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv2D(256, 4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv2D(512, 4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    output = keras.layers.Conv2D(1, 4, strides=1, padding='valid', use_bias=False)(x)
    return keras.Model(inputs=inputs, outputs=output, name=name)
