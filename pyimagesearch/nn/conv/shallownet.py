# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 23:17:09 2021

@author: DESKTOP
"""
# Importing the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from keras.layers import Dense
from tensorflow.keras import backend as K

class ShallowNet:

  @staticmethod
  def build(width, height, depth, classes):

    model = Sequential()
    inputShape = (height, width, depth)

    if K.image_data_format() == 'channels_first':
      inputShape = (depth, height, width)

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=inputShape))
    model.add(Activation("relu"))

    # Softmax Layer
    model.add(Flatten())
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model