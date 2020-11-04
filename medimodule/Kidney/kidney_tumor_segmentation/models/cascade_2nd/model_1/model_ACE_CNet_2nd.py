# coding: utf-8

import tensorflow as tf
from tensorflow.keras.models import Model
# from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Activation, Flatten, Lambda, LeakyReLU, Multiply, Reshape, ThresholdedReLU
from keras_contrib.layers import InstanceNormalization
from tensorflow.keras.layers import (Input, Conv3D, MaxPooling3D, UpSampling3D, ZeroPadding3D, Cropping3D, Conv3DTranspose,
                          GlobalAveragePooling3D, Concatenate, Convolution3D)
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import BatchNormalization, GaussianNoise
from tensorflow.keras.layers import concatenate, add, multiply
# from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import backend as K

smooth = 1.


def average_dice_coef(y_true, y_pred):
    y_pred = ThresholdedReLU(theta=0.5)(y_pred)
    loss = 0
    label_length = y_pred.get_shape().as_list()[-1]
    for num_label in range(label_length):
        y_true_f = K.flatten(y_true[..., num_label])
        y_pred_f = K.flatten(y_pred[..., num_label])
        intersection = K.sum(y_true_f * y_pred_f)
        loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return loss / label_length  # 1>= loss >0


def avg_dice_0(y_true, y_pred):
    y_pred = ThresholdedReLU(theta=0.5)(y_pred)
    loss = 0
    num_label = 0
    y_true_f = K.flatten(y_true[..., num_label])
    y_pred_f = K.flatten(y_pred[..., num_label])
    intersection = K.sum(y_true_f * y_pred_f)
    loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return loss  # 1>= loss >0


def avg_dice_1(y_true, y_pred):
    y_pred = ThresholdedReLU(theta=0.5)(y_pred)
    loss = 0
    num_label = 1
    y_true_f = K.flatten(y_true[..., num_label])
    y_pred_f = K.flatten(y_pred[..., num_label])
    intersection = K.sum(y_true_f * y_pred_f)
    loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return loss  # 1>= loss >0


def avg_dice_2(y_true, y_pred):
    y_pred = ThresholdedReLU(theta=0.5)(y_pred)
    loss = 0
    num_label = 2
    y_true_f = K.flatten(y_true[..., num_label])
    y_pred_f = K.flatten(y_pred[..., num_label])
    intersection = K.sum(y_true_f * y_pred_f)
    loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return loss  # 1>= loss >0


def average_dice_coef_loss(y_true, y_pred):
    return -average_dice_coef(y_true, y_pred)


# def ACE_CNet_2nd(input_shape, num_labels, axis=-1, base_filter=32, depth_size=4, se_res_block=True, se_ratio=16,
#                noise=0.1, last_relu=False, atten_gate=False):
#     def conv3d(layer_input, filters, axis=-1, se_res_block=True, se_ratio=16, down_sizing=True):
#         if down_sizing == True:
#             layer_input = MaxPooling3D(pool_size=(2, 2, 2))(layer_input)
#         d = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(layer_input)
#         d = InstanceNormalization(axis=axis)(d)
#         d = LeakyReLU(alpha=0.3)(d)
#         d = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(d)
#         d = InstanceNormalization(axis=axis)(d)
#         if se_res_block == True:
#             se = GlobalAveragePooling3D()(d)
#             se = Dense(filters // se_ratio, activation='relu')(se)
#             se = Dense(filters, activation='sigmoid')(se)
#             se = Reshape([1, 1, 1, filters])(se)
#             d = Multiply()([d, se])
#             shortcut = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(layer_input)
#             shortcut = InstanceNormalization(axis=axis)(shortcut)
#             d = add([d, shortcut])
#         d = LeakyReLU(alpha=0.3)(d)
#         return d
#
#     def deconv3d(layer_input, skip_input, filters, axis=-1, se_res_block=True, se_ratio=16, atten_gate=False):
#         if atten_gate == True:
#             gating = Conv3D(filters, (1, 1, 1), use_bias=False, padding='same')(layer_input)
#             gating = InstanceNormalization(axis=axis)(gating)
#             attention = Conv3D(filters, (2, 2, 2), strides=(2, 2, 2), use_bias=False, padding='valid')(skip_input)
#             attention = InstanceNormalization(axis=axis)(attention)
#             attention = add([gating, attention])
#             attention = Conv3D(1, (1, 1, 1), use_bias=False, padding='same', activation='sigmoid')(attention)
#             # attention = Lambda(resize_by_axis, arguments={'dim_1':skip_input.get_shape().as_list()[1],'dim_2':skip_input.get_shape().as_list()[2],'ax':3})(attention) # error when None dimension is feeded.
#             # attention = Lambda(resize_by_axis, arguments={'dim_1':skip_input.get_shape().as_list()[1],'dim_2':skip_input.get_shape().as_list()[3],'ax':2})(attention)
#             attention = ZeroPadding3D(((0, 1), (0, 1), (0, 1)))(attention)
#             attention = UpSampling3D((2, 2, 2))(attention)
#             attention = CropToConcat3D(mode='crop')([attention, skip_input])
#             attention = Lambda(lambda x: K.tile(x, [1, 1, 1, 1, filters]))(attention)
#             skip_input = multiply([skip_input, attention])
#
#         u1 = ZeroPadding3D(((0, 1), (0, 1), (0, 1)))(layer_input)
#         u1 = Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), use_bias=False, padding='same')(u1)
#         u1 = InstanceNormalization(axis=axis)(u1)
#         u1 = LeakyReLU(alpha=0.3)(u1)
#         u1 = CropToConcat3D()([u1, skip_input])
#         u2 = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(u1)
#         u2 = InstanceNormalization(axis=axis)(u2)
#         u2 = LeakyReLU(alpha=0.3)(u2)
#         u2 = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(u2)
#         u2 = InstanceNormalization(axis=axis)(u2)
#         if se_res_block == True:
#             se = GlobalAveragePooling3D()(u2)
#             se = Dense(filters // se_ratio, activation='relu')(se)
#             se = Dense(filters, activation='sigmoid')(se)
#             se = Reshape([1, 1, 1, filters])(se)
#             u2 = Multiply()([u2, se])
#             shortcut = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(u1)
#             shortcut = InstanceNormalization(axis=axis)(shortcut)
#             u2 = add([u2, shortcut])
#         u2 = LeakyReLU(alpha=0.3)(u2)
#         return u2
#
#     def CropToConcat3D(mode='concat'):
#         def crop_to_concat_3D(concat_layers, axis=-1):
#             bigger_input, smaller_input = concat_layers
#             bigger_shape, smaller_shape = tf.shape(bigger_input), \
#                                           tf.shape(smaller_input)
#             sh, sw, sd = smaller_shape[1], smaller_shape[2], smaller_shape[3]
#             bh, bw, bd = bigger_shape[1], bigger_shape[2], bigger_shape[3]
#             dh, dw, dd = bh - sh, bw - sw, bd - sd
#             cropped_to_smaller_input = bigger_input[:, :-dh,
#                                        :-dw,
#                                        :-dd, :]
#             if mode == 'concat':
#                 return K.concatenate([smaller_input, cropped_to_smaller_input], axis=axis)
#             elif mode == 'add':
#                 return smaller_input + cropped_to_smaller_input
#             elif mode == 'crop':
#                 return cropped_to_smaller_input
#
#         return Lambda(crop_to_concat_3D)
#
#     def resize_by_axis(image, dim_1, dim_2, ax):  # it is available only for 1 channel 3D
#         resized_list = []
#         unstack_img_depth_list = tf.unstack(image, axis=ax)
#         for i in unstack_img_depth_list:
#             resized_list.append(tf.image.resize_images(i, [dim_1, dim_2]))  # defaults to ResizeMethod.BILINEAR
#         stack_img = tf.stack(resized_list, axis=ax + 1)
#         return stack_img
#
#     input_img = Input(shape=input_shape, name='Input')
#     d0 = GaussianNoise(noise)(input_img)
#     d1 = Conv3D(base_filter, (3, 3, 3), use_bias=False, padding='same')(d0)
#     d1 = InstanceNormalization(axis=axis)(d1)
#     d1 = LeakyReLU(alpha=0.3)(d1)
#     d2 = conv3d(d1, base_filter * 2, se_res_block=se_res_block)
#     d3 = conv3d(d2, base_filter * 4, se_res_block=se_res_block)
#     d4 = conv3d(d3, base_filter * 8, se_res_block=se_res_block)
#
#     if depth_size == 4:
#         d5 = conv3d(d4, base_filter * 16, se_res_block=se_res_block)
#         u4 = deconv3d(d5, d4, base_filter * 8, se_res_block=se_res_block, atten_gate=atten_gate)
#         u3 = deconv3d(u4, d3, base_filter * 4, se_res_block=se_res_block, atten_gate=atten_gate)
#     elif depth_size == 3:
#         u3 = deconv3d(d4, d3, base_filter * 4, se_res_block=se_res_block, atten_gate=atten_gate)
#     else:
#         raise Exception('depth size must be 3 or 4. you put ', depth_size)
#
#     u2 = deconv3d(u3, d2, base_filter * 2, se_res_block=se_res_block, atten_gate=atten_gate)
#     u1 = ZeroPadding3D(((0, 1), (0, 1), (0, 1)))(u2)
#     u1 = Conv3DTranspose(base_filter, (2, 2, 2), strides=(2, 2, 2), use_bias=False, padding='same')(u1)
#     u1 = InstanceNormalization(axis=axis)(u1)
#     u1 = LeakyReLU(alpha=0.3)(u1)
#     u1 = CropToConcat3D()([u1, d1])
#     # output_img = Conv3D(num_labels, (3, 3, 3), use_bias=True, padding='same', activation='sigmoid', name='lastconv')(u1)
#     u1 = Conv3D(base_filter, (3, 3, 3), use_bias=False, padding='same')(u1)
#     u1 = InstanceNormalization(axis=axis)(u1)
#     u1 = LeakyReLU(alpha=0.3)(u1)
#     output_img = Conv3D(num_labels, kernel_size=1, strides=1, padding='same', activation='sigmoid', name='lastconv')(u1)
#     if last_relu == True:
#         output_img = ThresholdedReLU(theta=0.5)(output_img)
#     model = Model(inputs=input_img, outputs=output_img)
#     return models


def slicechannel_lamda(x, start, end):
    return x[:, :, :, start:end]


def _phase_shift_3D_wChannel(I, r, out_c):
    bsize, z, y, x, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, z, y, x, r, r, r, out_c))

    X = tf.transpose(X, (0, 1, 2, 3, 6, 5, 4, 7))  # bsize, z, y, x, 1, 1, 1
    X = tf.split(X, z, axis=1)  # z, [bsize, 1, y, x, r, r, r, out_c]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=3)  # bsize, y, x, z*r, r, r

    X = tf.split(X, y, axis=1)  # y, [bsize, x, z*r, r, r, out_c]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=3)  # bsize, x, z*r, y*r, r

    X = tf.split(X, x, axis=1)  # x, [bsize, z*r, y*r, r, out_c]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=3)  # bsize, z*r, y*r, x*r

    return tf.reshape(X, (bsize, z * r, y * r, x * r, out_c))


def subpixel_2_3D_wChannel(out_c, **kwargs):
    return Lambda(lambda x: _phase_shift_3D_wChannel(x, 2, out_c), **kwargs)


def subpixel_4_3D_wChannel(out_c, **kwargs):
    return Lambda(lambda x: _phase_shift_3D_wChannel(x, 4, out_c), **kwargs)


def subpixel_8_3D_wChannel(out_c, **kwargs):
    return Lambda(lambda x: _phase_shift_3D_wChannel(x, 8, out_c), **kwargs)


class SegMultiOrganNet:
    def __init__(self, input, network_name):
        self.concat_axis = -1
        self.out = getattr(self, network_name)(input)
        self.model = Model(inputs=input, outputs=self.out)

    def dense_block(self, input_tensor, nb_layers, growth_k):
        output_intermediate = self.bottleneck_layer(input_tensor=input_tensor, growth_k=growth_k)
        output_tensor = Concatenate(axis=self.concat_axis)([input_tensor, output_intermediate])

        for idx in range(1, nb_layers):
            output_intermediate = self.bottleneck_layer(output_tensor, growth_k=growth_k)
            output_tensor = Concatenate(axis=self.concat_axis)([output_tensor, output_intermediate])

        return output_tensor

    def bottleneck_layer(self, input_tensor, growth_k):
        out_bn_1 = InstanceNormalization(axis=self.concat_axis, center=True)(input_tensor)
        out_relu_1 = Activation('relu')(out_bn_1)

        out_conv_1 = Convolution3D(filters=growth_k * 4, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same',
                                   use_bias=True, kernel_initializer='he_normal',
                                   kernel_regularizer=None)(out_relu_1)

        out_bn_2 = InstanceNormalization(axis=self.concat_axis, center=True)(out_conv_1)
        out_relu_2 = Activation('relu')(out_bn_2)

        output_tensor = Convolution3D(filters=growth_k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                      use_bias=True, kernel_initializer='he_normal',
                                      kernel_regularizer=None)(out_relu_2)

        return output_tensor

    def transition_layer(self, input_tensor, theta=0.5):
        nb_channel = int(input_tensor.shape[-1])
        out_bn_1 = InstanceNormalization(axis=self.concat_axis, center=True)(input_tensor)
        out_conv_1 = Convolution3D(filters=int(nb_channel * theta), kernel_size=(1, 1, 1), strides=(1, 1, 1),
                                   padding='same', use_bias=True, kernel_initializer='he_normal',
                                   kernel_regularizer=None)(out_bn_1)
        return out_conv_1

    def last_label_layer(self, input_tensor):
        bn_1 = InstanceNormalization(axis=self.concat_axis, center=True)(input_tensor)
        relu_1 = Activation('relu')(bn_1)
        conv_1 = Convolution3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                               use_bias=True, kernel_initializer='he_normal',
                               kernel_regularizer=None)(relu_1)

        bn_2 = InstanceNormalization(axis=self.concat_axis, center=True)(conv_1)
        relu_2 = Activation('relu')(bn_2)
        conv_2 = Convolution3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                               use_bias=True, kernel_initializer='he_normal',
                               kernel_regularizer=None)(relu_2)
        return subpixel_8_3D_wChannel(out_c=1)(conv_2)

    def DenseNet_v3(self, input):
        # growth_k = 8
        growth_k = 4
        theta = 0.5
        init_conv_0 = Convolution3D(filters=growth_k * 2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                    use_bias=True, kernel_initializer='he_normal',
                                    kernel_regularizer=None)(input)  # 200 x 200 x 200
        init_bn_0 = InstanceNormalization(axis=self.concat_axis, center=False)(init_conv_0)
        init_relu_0 = Activation('relu')(init_bn_0)
        init_conv_1 = Convolution3D(filters=growth_k * 2, kernel_size=(7, 7, 7), strides=(2, 2, 2), padding='same',
                                    use_bias=True, kernel_initializer='he_normal',
                                    kernel_regularizer=None)(init_relu_0)  # 100 x 100 x 100

        nb_layers = 12
        denseblk_1 = self.dense_block(init_conv_1, nb_layers=nb_layers, growth_k=growth_k)
        transit_1 = self.transition_layer(denseblk_1, theta=theta)
        bsize, zz, yy, xx, c = transit_1.get_shape().as_list()
        transit_1 = Convolution3D(filters=c, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same',
                                  use_bias=True, kernel_initializer='he_normal',
                                  kernel_regularizer=None)(transit_1)  # 50 x 50 x50

        nb_layers = 24
        denseblk_2 = self.dense_block(transit_1, nb_layers=nb_layers, growth_k=growth_k)
        transit_2 = self.transition_layer(denseblk_2, theta=theta)
        bsize, zz, yy, xx, c = transit_2.get_shape().as_list()
        transit_2 = Convolution3D(filters=c, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same',
                                  use_bias=True, kernel_initializer='he_normal',
                                  kernel_regularizer=None)(transit_2)  # 25 x 25 x 25

        nb_layers = 18
        denseblk_3 = self.dense_block(transit_2, nb_layers=nb_layers, growth_k=growth_k)

        label_0 = self.last_label_layer(denseblk_3)
        label_1 = self.last_label_layer(denseblk_3)
        label_2 = self.last_label_layer(denseblk_3)
        concat_1 = Concatenate()([label_0, label_1, label_2, init_relu_0])
        last_conv_0 = Convolution3D(filters=growth_k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                    use_bias=True, kernel_initializer='he_normal',
                                    kernel_regularizer=None)(concat_1)  # 200 x 200 x 200
        last_bn_0 = InstanceNormalization(axis=self.concat_axis, center=False)(last_conv_0)
        last_relu_0 = Activation('relu')(last_bn_0)
        last_conv_1 = Convolution3D(filters=3, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                    use_bias=True, kernel_initializer='he_normal',
                                    kernel_regularizer=None)(last_relu_0)

        return last_conv_1
