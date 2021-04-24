# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 14:22:33 2021

@author: dhrv04@gmail.com
"""
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class MiniVGGNet:
    
    @staticmethod
    def build(width, height, depth, classes):
        
        model = Sequential()
        inputShape = (height, width, depth)
        # Batch Normalization operates over channels, so in order to normailize over we need to the axis to normalize over
        # ... setting chanDim = -1 implies that the index of the channel dimension is last in the input shape
        chanDim = -1
        
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
            
        # First Layer of Convolutional Network
        model.add(Conv2D(32, (3,3), padding='same', input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3,3), padding='same', input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))        
        model.add(Dropout(0.25))
        
        # Second Layer of Convolutional Network
        model.add(Conv2D(32, (3,3), padding='same', input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3,3), padding='same', input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))        
        model.add(Dropout(0.25))
        
        # FC layer
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        