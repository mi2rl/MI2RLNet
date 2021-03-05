import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import GlobalAveragePooling3D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import ZeroPadding3D
from tensorflow.keras.layers import UpSampling3D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv3DTranspose
from tensorflow.keras.layers import ThresholdedReLU
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.models import Model

import numpy as np
import SimpleITK as sitk
from typing import Tuple, Optional
from scipy.ndimage import label


def se_block(
    inputs: tf.Tensor,
    block_input: tf.Tensor,
    filters: int,
    se_ratio: int = 16
) -> tf.Tensor:

    se = GlobalAveragePooling3D()(inputs)
    se = Dense(filters // se_ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape([1, 1, 1, filters])(se)
    x = Multiply()([inputs, se])
    shortcut = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(block_input)
    shortcut = InstanceNormalization()(shortcut)
    x = Add()([x, shortcut])
    return x


def conv3d(
    inputs: tf.Tensor, 
    filters: int, 
    is_se_block: bool = True, 
    se_ratio: int = 16, 
    downsizing: bool = True,
    activation: bool = True,
    loop: int = 2
) -> tf.Tensor:

    if downsizing:
        inputs = MaxPooling3D(pool_size=(2, 2, 2))(inputs)

    x = inputs
    for i in range(loop):
        x = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(x)
        x = InstanceNormalization()(x)
        if is_se_block and i > 0:
            x = se_block(x, inputs, filters, se_ratio=se_ratio)
        if activation:
            x = LeakyReLU(alpha=0.3)(x)
            
    return x


def upconv3d(
    inputs: tf.Tensor, 
    skip_input: tf.Tensor, 
    filters: int, 
    is_se_block: bool = True, 
    se_ratio: bool = 16, 
    loop: int = 2
) -> tf.Tensor:

    x = ZeroPadding3D(((0, 1), (0, 1), (0, 1)))(inputs)
    x = Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), use_bias=False, padding='same')(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = _crop_concat()([x, skip_input])
    x = conv3d(x, filters, is_se_block=is_se_block, se_ratio=se_ratio, downsizing=False, loop=loop)
    return x


def _crop_concat(mode: str = 'concat') -> tf.keras.layers.Layer:
    def crop(concat_layers: tf.Tensor) -> tf.Tensor:
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


def KidneyTumorSeg(
    input_shape: Tuple[Optional[int], Optional[int], Optional[int], int], 
    num_labels: int, 
    base_filter: int = 32, 
    depth: int = 4, 
    se_res_block: bool = True, 
    se_ratio: int = 16,
    noise: float = 0.1, 
    last_relu: bool = False
) -> Model:

    img_input = Input(shape=input_shape)

    d0 = GaussianNoise(noise)(img_input)
    d1 = conv3d(d0, base_filter, is_se_block=False, se_ratio=se_ratio, downsizing=False, loop=1)
    d2 = conv3d(d1, base_filter*2, se_ratio=se_ratio)
    d3 = conv3d(d2, base_filter*4, se_ratio=se_ratio)
    d4 = conv3d(d3, base_filter*8, se_ratio=se_ratio)

    if depth == 4:
        d5 = conv3d(d4, base_filter*16, se_ratio=se_ratio)

        u4 = upconv3d(d5, d4, base_filter*8, se_ratio=se_ratio)
        u3 = upconv3d(u4, d3, base_filter*4, se_ratio=se_ratio)
    elif depth == 3:
        u3 = upconv3d(d4, d3, base_filter*4, se_ratio=se_ratio)
    else:
        raise Exception('Depth size must be 3 or 4. You put ', depth)

    u2 = upconv3d(u3, d2, base_filter*2, se_ratio=se_ratio)
    u1 = upconv3d(u2, d1, base_filter, is_se_block=False, se_ratio=se_ratio, loop=1)

    if num_labels == 1:
        img_output = Conv3D(num_labels, (1, 1, 1), strides=(1, 1, 1), padding='same', activation='sigmoid')(u1)
    else:
        img_output = Conv3D(num_labels, (1, 1, 1), strides=(1, 1, 1), padding='same', activation='softmax')(u1)

    model = Model(img_input, img_output)

    return model


class KidneyUtils:
    @staticmethod
    def resample_img_asdim(
        img: sitk.Image, 
        target_size_itkorder: Tuple[int, int, int], 
        interp: int = sitk.sitkLinear, 
        c_val: int = 0
    ) -> sitk.Image:

        img_size = img.GetSize()
        img_spacing = img.GetSpacing()
        target_spacing = ((img_size[0]) / target_size_itkorder[0] * img_spacing[0],
                          (img_size[1]) / target_size_itkorder[1] * img_spacing[1],
                          (img_size[2]) / target_size_itkorder[2] * img_spacing[2])

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img)
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetInterpolator(interp)
        resampler.SetSize(target_size_itkorder)
        resampler.SetDefaultPixelValue(c_val)

        return resampler.Execute(img)

    @staticmethod
    def transaxis(img: sitk.Image, dtype: np.dtype) -> sitk.Image:
        spacing = img.GetSpacing()

        img_raw = sitk.GetArrayFromImage(img)
        img_raw = np.transpose(img_raw, axes=[2, 1, 0])
        img_raw = img_raw[-1::-1, :, :]
        img_new = sitk.GetImageFromArray(img_raw.astype(dtype))
        img_new.SetSpacing(tuple(reversed(spacing)))
        img_new.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

        return img_new

    @staticmethod
    def normalize_vol(
        vol: np.ndarray, 
        norm_wind_lower: int, 
        norm_wind_upper: int
    ) -> np.ndarray:

        slope = norm_wind_upper - norm_wind_lower
        vol = vol - norm_wind_lower
        vol = vol / slope
        vol[np.where(vol < 0)] = 0
        vol[np.where(vol > 1)] = 1

        return vol

    @staticmethod
    def CCL_check_1ststg(vol: np.ndarray) -> np.ndarray:
        structure_label = np.ones(shape=(3, 3, 3))
        new_vol = np.array(vol)
        new_vol[np.where(new_vol > 0.5)] = 1
        _, max_num_compo = label(new_vol, structure=structure_label)
        if max_num_compo > 70:
            new_vol[:, :, :] = 0
        
        return new_vol

    @staticmethod
    def CCL_1ststg_post(vol: np.ndarray) -> np.ndarray:
        structure_label = np.ones(shape=(3, 3, 3))
        new_vol = np.zeros(shape=np.shape(vol)).astype(np.uint8)
        idx = 1
        vol_binary = np.zeros(np.shape(vol)).astype(np.uint16)
        vol_binary[np.where(vol == idx)] = 1

        labeled_vol, max_num_compo = label(vol_binary, structure=structure_label)
        volume_each_compo = [0]
        for idx_comp in range(1, max_num_compo + 1):
            volume_each_compo.append(len(np.where(labeled_vol == idx_comp)[0]))

        idx_for_maxvol = np.argmax(volume_each_compo)
        new_vol[np.where(labeled_vol == idx_for_maxvol)] = idx
        volume_each_compo[idx_for_maxvol] = 0
        idcs_first = np.where(labeled_vol == idx_for_maxvol)
        # one more time

        if max_num_compo > 1:
            idx_for_maxvol = np.argmax(volume_each_compo)
            idcs_second = np.where(labeled_vol == idx_for_maxvol)
            if len(idcs_second[0]) > len(idcs_first[0]) * 0.2:
                new_vol[np.where(labeled_vol == idx_for_maxvol)] = idx + 1

        return new_vol

    @staticmethod
    def CCL(vol: np.ndarray, num_labels: int) -> np.ndarray:
        structure_label = np.ones(shape=(3, 3, 3))
        new_vol = np.zeros(shape=np.shape(vol)).astype(np.uint8)
        for idx in range(1, num_labels):
            # print('CCL_processing [%d]' % idx)
            vol_binary = np.zeros(np.shape(vol)).astype(np.uint16)
            vol_binary[np.where(vol == idx)] = 1

            labeled_vol, max_num_compo = label(vol_binary, structure=structure_label)
            # print('max_num_compo: %d' % max_num_compo)
            if max_num_compo > 70:
                # print('continue')
                continue

            volume_each_compo = [0]
            for idx_comp in range(1, max_num_compo + 1):
                volume_each_compo.append(len(np.where(labeled_vol == idx_comp)[0]))

            idx_for_maxvol = np.argmax(volume_each_compo)
            new_vol[np.where(labeled_vol == idx_for_maxvol)] = idx

        return new_vol