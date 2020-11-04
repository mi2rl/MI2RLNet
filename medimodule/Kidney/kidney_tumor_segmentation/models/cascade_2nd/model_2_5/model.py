# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import ZeroPadding3D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv3DTranspose
from tensorflow.keras.layers import GlobalAveragePooling3D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Add

from tensorflow.keras.models import Model

from keras_contrib.layers import InstanceNormalization

from .loss import *
from .metrics import *


class MyModel():
    def __init__(self,
                 model,
                 input_shape,
                 lossfn='dice',
                 classes=3,
                 axis=-1,
                 noise=0.1,
                 depth=4,
                 base_filter=32):

        self.model = model
        self.input_shape = input_shape
        self.lossfn = lossfn
        self.classes = classes
        self.axis = axis
        self.noise = noise
        self.depth = depth
        self.base_filter = base_filter
        
        self.se_ratio = 16

        self.mymodel = self.Unet3D()
        
    def _set_loss(self):
        if self.lossfn == 'dice':
            self.loss = dice_loss
        elif self.lossfn == 'focaldice':
            self.loss = focal_dice_loss

    def _set_optimizer(self, lr):
        from tf.keras.optimizers import Adam
        self.optimizer = Adam(lr=lr)

    def _set_metrics(self):
        self.metrics = [dice, kidney_dice, tumor_dice, iou]

    def compile(self, lr):
        self._set_loss()
        self._set_optimizer(lr)
        self._set_metrics()

        self.mymodel.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics)        

    def Unet3D(self):
        img_input = Input(shape=self.input_shape)

        d0 = GaussianNoise(self.noise)(img_input)
        d1 = self._conv3d(d0, self.base_filter, se_block=False, se_ratio=self.se_ratio, downsizing=False, loop=1)
        d2 = self._conv3d(d1, self.base_filter*2, se_ratio=self.se_ratio)
        d3 = self._conv3d(d2, self.base_filter*4, se_ratio=self.se_ratio)
        d4 = self._conv3d(d3, self.base_filter*8, se_ratio=self.se_ratio)

        if self.depth == 4:
            d5 = self._conv3d(d4, self.base_filter*16, se_ratio=self.se_ratio)

            u4 = self._upconv3d(d5, d4, self.base_filter*8, se_ratio=self.se_ratio)
            u3 = self._upconv3d(u4, d3, self.base_filter*4, se_ratio=self.se_ratio)
        elif self.depth == 3:
            u3 = self._upconv3d(d4, d3, self.base_filter*4, se_ratio=self.se_ratio)
        else:
            raise Exception('Depth size must be 3 or 4. You put ', self.depth_size)

        u2 = self._upconv3d(u3, d2, self.base_filter*2, se_ratio=self.se_ratio)
        u1 = self._upconv3d(u2, d1, self.base_filter, se_block=False, se_ratio=self.se_ratio, loop=1)

        if self.classes == 1:
            img_output = Conv3D(self.classes, (1, 1, 1), strides=(1, 1, 1), padding='same', activation='sigmoid')(u1)
        else:
            img_output = Conv3D(self.classes, (1, 1, 1), strides=(1, 1, 1), padding='same', activation='softmax')(u1)

        model = Model(img_input, img_output, name=self.model)

        return model
        
    def _se_block(self, inputs, block_input, filters, se_ratio=16):
        se = GlobalAveragePooling3D()(inputs)
        se = Dense(filters//se_ratio, activation='relu')(se)
        se = Dense(filters, activation='sigmoid')(se)
        se = Reshape([1, 1, 1, filters])(se)
        x = Multiply()([inputs, se])
        shortcut = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(block_input)
        shortcut = self._norm(shortcut)
        x = Add()([x, shortcut])
        return x

    def _conv3d(self, inputs, filters, se_block=True, se_ratio=16, downsizing=True, activation=True, loop=2):
        if downsizing:
            inputs = MaxPooling3D(pool_size=(2, 2, 2))(inputs)
        
        x = inputs
        for i in range(loop):
            x = Conv3D(filters, (3, 3, 3), strides=(1, 1, 1), use_bias=False, padding='same')(x)
            x = self._norm(x)
            if se_block and i > 0:
                x = self._se_block(x, inputs, filters, se_ratio=se_ratio)
            if activation:
                x = self._activation(x)
        return x

    def _upconv3d(self, inputs, skip_input, filters, se_block=True, se_ratio=16, loop=2):
        x = ZeroPadding3D(((0, 1), (0, 1), (0, 1)))(inputs)
        x = Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), use_bias=False, padding='same')(x)
        x = self._norm(x)
        x = self._activation(x)
        x = self._crop_concat()([x, skip_input])
        x = self._conv3d(x, filters, se_block=se_block, se_ratio=se_ratio, downsizing=False, loop=loop)
        return x

    def _norm(self, inputs, axis=-1):
        return InstanceNormalization(axis=axis)(inputs)

    def _activation(self, inputs):
        return LeakyReLU(alpha=0.3)(inputs)

    def _crop_concat(self, mode='concat'):
        def crop(concat_layers):
            big, small = concat_layers
            big_shape, small_shape = tf.shape(big), tf.shape(small)
            sh, sw, sd = small_shape[1], small_shape[2], small_shape[3]
            bh, bw, bd = big_shape[1], big_shape[2] ,big_shape[3]
            dh, dw, dd = bh-sh, bw-sw, bd-sd
            big_crop = big[:,:-dh,:-dw,:-dd,:]
            
            if mode == 'concat':
                return K.concatenate([small, big_crop], axis=-1)
            elif mode == 'add':
                return small + big_crop
            elif mode == 'crop':
                return big_crop
        return Lambda(crop)


if __name__ == "__main__":
    model = MyModel(model='seresattention-base',
                    input_shape=(32, 32, 32, 1),
                    depth=4)

    model.mymodel.summary()
    # from keras.utils import plot_model
    # plot_model(model.mymodel, to_file='./model.png')
