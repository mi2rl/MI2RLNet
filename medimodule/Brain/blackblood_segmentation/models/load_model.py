import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool3D
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import ZeroPadding3D
from tensorflow.keras.layers import Conv3DTranspose
from tensorflow.keras.layers import Lambda
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GaussianNoise


def build_blackblood_segmentation(
    weight_path: str,
    base_filter: int=32):

    def conv3d(inputs, filters, downsizing=True, loop=2):
        if downsizing:
            inputs = MaxPool3D(pool_size=(2, 2, 2))(inputs)
        x = inputs
        for _ in range(loop):
            x = Conv3D(filters, (3, 3, 3), strides=(1, 1, 1), use_bias=False, padding='same')(x)
            x = InstanceNormalization()(x)
            x = Activation('relu')(x)
        return x

    def upconv3d(inputs, skip_input, filters, loop=2):
        def _crop_concat():
            def crop(concat_layers):
                big, small = concat_layers
                big_shape, small_shape = tf.shape(big), tf.shape(small)
                sh, sw, sd = small_shape[1], small_shape[2], small_shape[3]
                bh, bw, bd = big_shape[1], big_shape[2] ,big_shape[3]
                dh, dw, dd = bh-sh, bw-sw, bd-sd
                big_crop = big[:,:-dh,:-dw,:-dd,:]
                return K.concatenate([small, big_crop], axis=-1)
            return Lambda(crop)


        x = ZeroPadding3D(((0, 1), (0, 1), (0, 1)))(inputs)
        x = Conv3DTranspose(filters, (2 ,2, 2), strides=(2, 2, 2), use_bias=False, padding='same')(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)

        x = _crop_concat()([x, skip_input])
        x = conv3d(x, filters, downsizing=False, loop=loop)
        return x

    img_input = Input(shape=(None, None, None, 1))
    d0 = GaussianNoise(0.1)(img_input)
    d1 = conv3d(d0, base_filter, downsizing=False, loop=1)
    d2 = conv3d(d1, base_filter*2, loop=2)
    d3 = conv3d(d2, base_filter*4, loop=2)
    d4 = conv3d(d3, base_filter*8, loop=2)
    d5 = conv3d(d4, base_filter*16, loop=2)

    u4 = upconv3d(d5, d4, base_filter*8)
    u3 = upconv3d(u4, d3, base_filter*4)
    u2 = upconv3d(u3, d2, base_filter*2)
    u1 = upconv3d(u2, d1, base_filter, loop=1)
    img_output = Conv3D(2, (1, 1, 1), strides=(1, 1, 1), padding='same', activation='softmax')(u1)

    model = Model(img_input, img_output, name='unet')
    model.load_weights(weight_path)
    return model

