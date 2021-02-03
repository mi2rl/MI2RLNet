import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization

# From model_2_5
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Lambda, LeakyReLU, Multiply, Reshape, ThresholdedReLU, Add



def ACE_CNet(input_shape, num_labels, axis=-1, base_filter=32, depth_size=4, se_res_block=True, se_ratio=16,noise=0.1, last_relu=False, atten_gate=False):
    def conv3d(layer_input, filters, axis=-1, se_res_block=True, se_ratio=16, down_sizing=True):
        if down_sizing == True:
            layer_input = MaxPooling3D(pool_size=(2, 2, 2))(layer_input)
        d = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(layer_input)
        d = InstanceNormalization(axis=axis)(d)
        d = LeakyReLU(alpha=0.3)(d)
        d = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(d)
        d = InstanceNormalization(axis=axis)(d)
        if se_res_block == True:
            se = GlobalAveragePooling3D()(d)
            se = Dense(filters // se_ratio, activation='relu')(se)
            se = Dense(filters, activation='sigmoid')(se)
            se = Reshape([1, 1, 1, filters])(se)
            d = Multiply()([d, se])
            shortcut = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(layer_input)
            shortcut = InstanceNormalization(axis=axis)(shortcut)
            d = Add()([d, shortcut])
        d = LeakyReLU(alpha=0.3)(d)
        return d

    def deconv3d(layer_input, skip_input, filters, axis=-1, se_res_block=True, se_ratio=16, atten_gate=False):
        if atten_gate == True:
            gating = Conv3D(filters, (1, 1, 1), use_bias=False, padding='same')(layer_input)
            gating = InstanceNormalization(axis=axis)(gating)
            attention = Conv3D(filters, (2, 2, 2), strides=(2, 2, 2), use_bias=False, padding='valid')(skip_input)
            attention = InstanceNormalization(axis=axis)(attention)
            attention = Add()([gating, attention])
            attention = Conv3D(1, (1, 1, 1), use_bias=False, padding='same', activation='sigmoid')(attention)
            # attention = Lambda(resize_by_axis, arguments={'dim_1':skip_input.get_shape().as_list()[1],'dim_2':skip_input.get_shape().as_list()[2],'ax':3})(attention) # error when None dimension is feeded.
            # attention = Lambda(resize_by_axis, arguments={'dim_1':skip_input.get_shape().as_list()[1],'dim_2':skip_input.get_shape().as_list()[3],'ax':2})(attention)
            attention = ZeroPadding3D(((0, 1), (0, 1), (0, 1)))(attention)
            attention = UpSampling3D((2, 2, 2))(attention)
            attention = CropToConcat3D(mode='crop')([attention, skip_input])
            attention = Lambda(lambda x: K.tile(x, [1, 1, 1, 1, filters]))(attention)
            skip_input = multiply([skip_input, attention])

        u1 = ZeroPadding3D(((0, 1), (0, 1), (0, 1)))(layer_input)
        u1 = Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), use_bias=False, padding='same')(u1)
        u1 = InstanceNormalization(axis=axis)(u1)
        u1 = LeakyReLU(alpha=0.3)(u1)
        u1 = CropToConcat3D()([u1, skip_input])
        u2 = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(u1)
        u2 = InstanceNormalization(axis=axis)(u2)
        u2 = LeakyReLU(alpha=0.3)(u2)
        u2 = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(u2)
        u2 = InstanceNormalization(axis=axis)(u2)
        if se_res_block == True:
            se = GlobalAveragePooling3D()(u2)
            se = Dense(filters // se_ratio, activation='relu')(se)
            se = Dense(filters, activation='sigmoid')(se)
            se = Reshape([1, 1, 1, filters])(se)
            u2 = Multiply()([u2, se])
            shortcut = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(u1)
            shortcut = InstanceNormalization(axis=axis)(shortcut)
            u2 = Add()([u2, shortcut])
        u2 = LeakyReLU(alpha=0.3)(u2)
        return u2

    def CropToConcat3D(mode='concat'):
        def crop_to_concat_3D(concat_layers, axis=-1):
            bigger_input, smaller_input = concat_layers
            bigger_shape, smaller_shape = tf.shape(bigger_input), \
                                          tf.shape(smaller_input)
            sh, sw, sd = smaller_shape[1], smaller_shape[2], smaller_shape[3]
            bh, bw, bd = bigger_shape[1], bigger_shape[2], bigger_shape[3]
            dh, dw, dd = bh - sh, bw - sw, bd - sd
            cropped_to_smaller_input = bigger_input[:, :-dh,
                                       :-dw,
                                       :-dd, :]
            if mode == 'concat':
                return K.concatenate([smaller_input, cropped_to_smaller_input], axis=axis)
            elif mode == 'add':
                return smaller_input + cropped_to_smaller_input
            elif mode == 'crop':
                return cropped_to_smaller_input

        return Lambda(crop_to_concat_3D)

    def resize_by_axis(image, dim_1, dim_2, ax):  # it is available only for 1 channel 3D
        resized_list = []
        unstack_img_depth_list = tf.unstack(image, axis=ax)
        for i in unstack_img_depth_list:
            resized_list.append(tf.image.resize_images(i, [dim_1, dim_2]))  # defaults to ResizeMethod.BILINEAR
        stack_img = tf.stack(resized_list, axis=ax + 1)
        return stack_img

    input_img = Input(shape=input_shape, name='Input')
    d0 = GaussianNoise(noise)(input_img)
    d1 = Conv3D(base_filter, (3, 3, 3), use_bias=False, padding='same')(d0)
    d1 = InstanceNormalization(axis=axis)(d1)
    d1 = LeakyReLU(alpha=0.3)(d1)
    d2 = conv3d(d1, base_filter * 2, se_res_block=se_res_block)
    d3 = conv3d(d2, base_filter * 4, se_res_block=se_res_block)
    d4 = conv3d(d3, base_filter * 8, se_res_block=se_res_block)

    if depth_size == 4:
        d5 = conv3d(d4, base_filter * 16, se_res_block=se_res_block)
        u4 = deconv3d(d5, d4, base_filter * 8, se_res_block=se_res_block, atten_gate=atten_gate)
        u3 = deconv3d(u4, d3, base_filter * 4, se_res_block=se_res_block, atten_gate=atten_gate)
    elif depth_size == 3:
        u3 = deconv3d(d4, d3, base_filter * 4, se_res_block=se_res_block, atten_gate=atten_gate)
    else:
        raise Exception('depth size must be 3 or 4. you put ', depth_size)

    u2 = deconv3d(u3, d2, base_filter * 2, se_res_block=se_res_block, atten_gate=atten_gate)
    u1 = ZeroPadding3D(((0, 1), (0, 1), (0, 1)))(u2)
    u1 = Conv3DTranspose(base_filter, (2, 2, 2), strides=(2, 2, 2), use_bias=False, padding='same')(u1)
    u1 = InstanceNormalization(axis=axis)(u1)
    u1 = LeakyReLU(alpha=0.3)(u1)
    u1 = CropToConcat3D()([u1, d1])
    u1 = Conv3D(base_filter, (3, 3, 3), use_bias=False, padding='same')(u1)
    u1 = InstanceNormalization(axis=axis)(u1)
    u1 = LeakyReLU(alpha=0.3)(u1)
    output_img = Conv3D(num_labels, kernel_size=1, strides=1, padding='same', activation='sigmoid')(u1)
    if last_relu == True:
        output_img = ThresholdedReLU(theta=0.5)(output_img)
    model = Model(inputs=input_img, outputs=output_img)
    return model
