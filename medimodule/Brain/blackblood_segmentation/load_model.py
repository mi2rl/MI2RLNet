
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
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import ThresholdedReLU
from tensorflow.keras.layers import GlobalAveragePooling3D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import UpSampling3D
from tensorflow_addons.layers import InstanceNormalization

from tensorflow.keras.models import Model


'''
convolution layers
'''

def build_blackblood_segmentation(
    weight_path: str,
    base_filter = 32,
    noise = 0.1,
    num_se = 0,
    norm = "in",
    activation = 'relu',
    skip = 'unet',
    se_ratio = 16,
    bottom_up = 'unet',
    top_down = 'unet',
    last_relu = 0,
    conv_loop = 2
    ):

    def conv3d(inputs, filters, mode='unet', downsizing=True, loop=2):
        if downsizing:
            if downsizing == 'pooling':
                inputs = MaxPooling3D(pool_size=(2, 2, 2))(inputs)
            elif downsizing == 'conv':
                inputs = Conv3D(filters, (3, 3, 3), strides=(2, 2, 2), use_bias=False, padding='same')(inputs)
                inputs = _norm(inputs)
                inputs = _activation(inputs)

        x = inputs
        if 'unet' in mode:
            x = _base_block(x, filters, loop=loop)  # conv + norm
        elif 'res' in mode:
            x = _residual_block(x, filters, mode=mode, loop=loop)
        else:
            pass

        return x

    def upconv3d(inputs, skip_input, filters, mode='unet', loop=2):
        if skip == 'attention':
            skip_input = _atten_gate(inputs, skip_input, filters)

        x = ZeroPadding3D(((0, 1), (0, 1), (0, 1)))(inputs)
        x = Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), use_bias=False, padding='same')(x)
        x = _norm(x)
        x = _activation(x)
        x = _crop_concat()([x, skip_input])
        x = conv3d(x, filters, mode=mode, downsizing=False, loop=loop)
        return x

    '''
    detail convolution layers
    '''

    def _base_block(inputs, filters, mode='unet', loop=2):
        x = inputs
        for i in range(loop):
            x = Conv3D(filters, (3, 3, 3), strides=(1, 1, 1), use_bias=False, padding='same')(x)
            x = _norm(x)
            if 'se' in mode and i >= loop - num_se:
                x = _se_block(x, filters, mode=mode)
            x = _activation(x)
        return x

    def _residual_block(inputs, filters, mode='res', loop=2):
        # res : base residual network
        # resbot : bottleneck residual network
        # resbotwhole : whole skip connection
        x = inputs
        for i in range(loop):
            if 'bot' in mode:
                # bottleneck
                # conv 1x1x(c/4) -> conv 3x3x(c/4) -> conv 1x1xc
                x = Conv3D(filters // 4, (1, 1, 1), strides=(1, 1, 1), use_bias=False, padding='same')(x)
                x = _norm(x)
                x = _activation(x)
                x = Conv3D(filters // 4, (3, 3, 3), strides=(1, 1, 1), use_bias=False, padding='same')(x)
                x = _norm(x)
                x = _activation(x)
                x = Conv3D(filters, (1, 1, 1), strides=(1, 1, 1), use_bias=False, padding='same')(x)
                x = _norm(x)
            else:
                x = Conv3D(filters, (3, 3, 3), strides=(1, 1, 1), use_bias=False, padding='same')(x)
                x = _norm(x)

            if 'whole' in mode:
                if i >= loop - 1:
                    if 'se' in mode:
                        x = _se_block(x, filters, mode=mode)

                    inputs = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(inputs)
                    inputs = _norm(inputs)
                    x = Add()([x, inputs])

                x = _activation(x)

            else:
                if 'se' in mode and i >= loop - num_se:
                    x = _se_block(x, filters, mode=mode)

                inputs = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(inputs)
                inputs = _norm(inputs)
                x = Add()([x, inputs])

                x = _activation(x)
                inputs = x

        return x



    def _se_block(inputs, filters, mode='se'):
        x = GlobalAveragePooling3D()(inputs)
        x = Dense(filters // se_ratio, activation='relu')(x)
        x = Dense(filters, activation='sigmoid')(x)
        x = Reshape([1, 1, 1, filters])(x)
        x = Multiply()([inputs, x])
        return x

    def _atten_gate(inputs, skip_input, filters):
        def __expend_as(tensor, rep):
            my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=4), arguments={'repnum': rep})(
                tensor)
            return my_repeat

        gating = Conv3D(K.int_shape(inputs)[-1], (1, 1, 1), use_bias=False, padding='same')(inputs)
        gating = _norm(gating)
        shape_skip = K.int_shape(skip)
        shape_gating = K.int_shape(gating)

        #
        theta = Conv3D(filters, (2, 2, 2), strides=(2, 2, 2), use_bias=False, padding='same')(skip)
        theta = _norm(theta)
        shape_theta = K.int_shape(theta)

        phi = Conv3D(filters, (1, 1, 1), use_bias=False, padding='same')(gating)
        phi = Conv3DTranspose(filters, (3, 3, 3),
                              strides=(shape_theta[1] // shape_gating[1], shape_theta[2] // shape_gating[2],
                                       shape_theta[3] // shape_gating[3]),
                              padding='same')(phi)

        add_xg = Add()([phi, theta])
        act_xg = Activation(activation='relu')(add_xg)
        psi = Conv3D(1, (1, 1, 1), use_bias=False, padding='same')(act_xg)
        sigmoid_xg = Activation(activation='sigmoid')(psi)
        shape_sigmoid = K.int_shape(sigmoid_xg)

        upsample_psi = UpSampling3D(size=(
        shape_skip[1] // shape_sigmoid[1], shape_skip[2] // shape_sigmoid[2], shape_skip[3] // shape_sigmoid[3]))(
            sigmoid_xg)
        upsample_psi = __expend_as(upsample_psi, shape_skip[3])
        result = Multiply()([upsample_psi, skip])
        result = Conv3D(shape_skip[3], (1, 1, 1), padding='same')(result)
        result = _norm(result)
        return result

    def _norm(inputs, axis=-1):
        if norm == 'bn':
            return BatchNormalization(axis=axis)(inputs)
        elif norm == 'in':
            return InstanceNormalization(axis=axis)(inputs)

    def _activation(inputs):
        if activation == 'relu':
            return Activation('relu')(inputs)
        elif activation == 'leakyrelu':
            return LeakyReLU(alpha=0.3)(inputs)

    def _crop_concat(mode='concat'):
        def crop(concat_layers):
            big, small = concat_layers
            big_shape, small_shape = tf.shape(big), tf.shape(small)
            sh, sw, sd = small_shape[1], small_shape[2], small_shape[3]
            bh, bw, bd = big_shape[1], big_shape[2], big_shape[3]
            dh, dw, dd = bh - sh, bw - sw, bd - sd
            big_crop = big[:, :-dh, :-dw, :-dd, :]

            if mode == 'concat':
                return K.concatenate([small, big_crop], axis=-1)
            elif mode == 'add':
                return small + big_crop
            elif mode == 'crop':
                return big_crop

        return Lambda(crop)


    img_input = Input(shape=(None, None, None, 1))
    d0 = GaussianNoise(noise)(img_input)
    d1 = conv3d(d0, base_filter, mode='unet', downsizing=False, loop=1)
    d2 = conv3d(d1, base_filter * 2, mode=bottom_up, loop=conv_loop)
    d3 = conv3d(d2, base_filter * 4, mode=bottom_up, loop=conv_loop)
    d4 = conv3d(d3, base_filter * 8, mode=bottom_up, loop=conv_loop)

    d5 = conv3d(d4, base_filter * 16, mode=bottom_up, loop=conv_loop)

    u4 = upconv3d(d5, d4, base_filter * 8, mode=top_down)
    u3 = upconv3d(u4, d3, base_filter * 4, mode=top_down)

    u2 = upconv3d(u3, d2, base_filter * 2, mode=top_down)
    u1 = upconv3d(u2, d1, base_filter, mode='unet', loop=1)

    img_output = Conv3D(2, (1, 1, 1), strides=(1, 1, 1), padding='same', activation='softmax')(u1)

    if last_relu == True:
        img_output = ThresholdedReLU(theta=0.5)(img_output)

    model = Model(img_input, img_output, name='unet')
    model.load_weights(weight_path)

    return model

