# Import the necessary packages
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K


class Resnet:

    @staticmethod
    def residual_module(data, K, stride, chanDim, red=False, reg=0.0001, bnEps=2e-5, bnMom=0.9):

        # Initializing the shortcut module as input of the data
        shortcut = data

        # The first block in the Resnet module are the 1x1 conv's
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)

        # the second block of the resnet module are the 3x3 CONV's
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        act2 = Activation('relu')(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding='same', use_bias=False,
                       kernel_regularizer=l2(reg))(act2)

        # the third block of module are 1x1 convs
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        act3 = Activation('relu')(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        # if we want to reduce the spatial size, then apply conv layer to the shortcut
        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)

        # Adding the together the shortcut and the final Conv
        x = add([conv3, shortcut])

        return x

    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg=0.0001, bnEps=2e-5, bnMom=0.9, dataset='cifar'):

        # Input dimension order
        inputShape = (height, width, depth)
        chanDim = -1

        # changing the dimension input if we are using channels first
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
            chanDim = -1

        # Set the input and apply BN
        inputs = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(inputs)

        # check if we are utilizing the cifar dataset
        if dataset == "cifar":

            # Applying the single conv layer
            x = Conv2D(filters[0], (3, 3), use_bias=False, padding='same', kernel_regularizer=l2(reg))(x)

            # Looping over the number of stages
        for i in range(0, len(stages)):

                # Initializing the strides and apply the residual module used to reduce the spatial size of the input volume
            stride = (1, 1) if i == 0 else (2, 2)
            x = Resnet.residual_module(x, filters[i+1], stride, chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

            # Looping over the number of layers in the stage
            for j in range(0, stages[i] - 1):

                # Applying a resnet module
                x = Resnet.residual_module(x, filters[i+1], (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)

        # Applying BN => ACT => POOL
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        x = Activation('relu')(x)
        x = AveragePooling2D((8, 8))(x)

        # Softmax classifier
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)

        # Creating the model
        model = Model(inputs, x, name='resnet')

        return model




































