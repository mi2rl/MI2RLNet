import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model


def LungSeg() -> Model:
    data = Input(shape=(1024, 1024, 1))
    conv1 = Conv2D(32, 3, padding='same', activation='relu')(data)
    conv1 = Conv2D(32, 3, padding='same', activation='relu')(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, padding='same', activation='relu')(pool1)
    conv2 = Conv2D(64, 3, padding='same', activation='relu')(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, 3, padding='same', activation='relu')(pool2)
    conv3 = Conv2D(64, 3, padding='same', activation='relu')(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, 3, padding='same', activation='relu')(pool3)
    conv4 = Conv2D(128, 3, padding='same', activation='relu')(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, 3, padding='same', activation='relu')(pool4)

    up1 = UpSampling2D(size=(2, 2))(conv5)
    conv6 = Conv2D(256, 3, padding='same', activation='relu')(up1)
    conv6 = Conv2D(256, 3, padding='same', activation='relu')(conv6)
    merged1 = Concatenate()([conv4, conv6])
    conv6 = Conv2D(256, 3, padding='same', activation='relu')(merged1)

    up2 = UpSampling2D(size=(2, 2))(conv6)
    conv7 = Conv2D(256, 3, padding='same', activation='relu')(up2)
    conv7 = Conv2D(256, 3, padding='same', activation='relu')(conv7)
    merged2 = Concatenate()([conv3, conv7])
    conv7 = Conv2D(256, 3, padding='same', activation='relu')(merged2)

    up3 = UpSampling2D(size=(2, 2))(conv7)
    conv8 = Conv2D(128, 3, padding='same', activation='relu')(up3)
    conv8 = Conv2D(128, 3, padding='same', activation='relu')(conv8)
    merged3 = Concatenate()([conv2, conv8])
    conv8 = Conv2D(128, 3, padding='same', activation='relu')(merged3)

    up4 = UpSampling2D(size=(2, 2))(conv8)
    conv9 = Conv2D(64, 3, padding='same', activation='relu')(up4)
    conv9 = Conv2D(64, 3, padding='same', activation='relu')(conv9)
    merged4 = Concatenate()([conv1, conv9])
    conv9 = Conv2D(64, 3, padding='same', activation='relu')(merged4)

    output = Conv2D(1, 3, padding='same', activation='sigmoid')(conv9)

    model = Model(data, output)

    return model