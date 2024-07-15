import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Conv2DTranspose
from keras.layers import concatenate
import numpy as np
import pandas as pd

def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):    # create one block of unet`s encoder
  conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
  conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)

  if dropout_prob > 0:
    conv = Dropout(dropout_prob)(conv)

  if max_pooling:
    next_layer = MaxPooling2D(pool_size=(2, 2))(conv)
  else:
    next_layer = conv

  skip_connection = conv
  return next_layer, skip_connection

def upsampling_block(expansive_input, contractive_input, n_filters=32):      # create one block of unet`s decoder
  up = Conv2DTranspose(n_filters, 3, strides=2, padding='same')(expansive_input)
  merge = concatenate([up, contractive_input], axis=3)
  conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge)
  conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
  return conv

def unet_model(input_size=(128, 128, 3), n_filters=32, n_classes=1, drop_prob=0.3):    # create whole unet-model
  inputs = Input(input_size)
  # encoder
  cblock1 = conv_block(inputs, n_filters)
  cblock2 = conv_block(cblock1[0], 2*n_filters)
  cblock3 = conv_block(cblock2[0], 4*n_filters)
  cblock4 = conv_block(cblock3[0], 8*n_filters, dropout_prob=drop_prob)
  cblock5 = conv_block(cblock4[0], 16*n_filters, dropout_prob=drop_prob, max_pooling=False)   # bottleneck
  # decoder
  ublock6 = upsampling_block(cblock5[0], cblock4[1],  8*n_filters)
  ublock7 = upsampling_block(ublock6, cblock3[1],  4*n_filters)
  ublock8 = upsampling_block(ublock7, cblock2[1],  2*n_filters)
  ublock9 = upsampling_block(ublock8, cblock1[1],  n_filters)
  conv9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ublock9)
  conv10 = Conv2D(n_classes, 1, padding='same')(conv9)
  model = tf.keras.Model(inputs=inputs, outputs=conv10)
  return model
