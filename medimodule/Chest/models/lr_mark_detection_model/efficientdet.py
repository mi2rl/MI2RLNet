import numpy as np
from functools import reduce

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model

from . import *
from .efficientnet import *
from .anchors import anchors_for_shape


def DepthwiseConvBlock(kernel_size, strides, name, freeze_bn=False):
    f1 = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same',
                                use_bias=False, name='{}_dconv'.format(name))
    f2 = BatchNormalization(freeze=freeze_bn, name='{}_bn'.format(name))
    f3 = ReLU(name='{}_relu'.format(name))
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2, f3))


def ConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    f1 = Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                       use_bias=False, name='{}_conv'.format(name))
    f2 = BatchNormalization(freeze=freeze_bn, name='{}_bn'.format(name))
    f3 = ReLU(name='{}_relu'.format(name))
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2, f3))


def build_BiFPN(features, num_channels, id, freeze_bn=False):
    if id == 0:
        _, _, C3, C4, C5 = features
        P3_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P3'.format(id))(
            C3)
        P4_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P4'.format(id))(
            C4)
        P5_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P5'.format(id))(
            C5)
        P6_in = ConvBlock(num_channels, kernel_size=3, strides=2, freeze_bn=freeze_bn, name='BiFPN_{}_P6'.format(id))(
            C5)
        P7_in = ConvBlock(num_channels, kernel_size=3, strides=2, freeze_bn=freeze_bn, name='BiFPN_{}_P7'.format(id))(
            P6_in)
    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features
        P3_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P3'.format(id))(
            P3_in)
        P4_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P4'.format(id))(
            P4_in)
        P5_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P5'.format(id))(
            P5_in)
        P6_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P6'.format(id))(
            P6_in)
        P7_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P7'.format(id))(
            P7_in)

    # upsample
    P7_U = UpSampling2D()(P7_in)
    P6_td = Add()([P7_U, P6_in])
    P6_td = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_U_P6'.format(id))(P6_td)
    P6_U = UpSampling2D()(P6_td)
    P5_td = Add()([P6_U, P5_in])
    P5_td = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_U_P5'.format(id))(P5_td)
    P5_U = UpSampling2D()(P5_td)
    P4_td = Add()([P5_U, P4_in])
    P4_td = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_U_P4'.format(id))(P4_td)
    P4_U = UpSampling2D()(P4_td)
    P3_out = Add()([P4_U, P3_in])
    P3_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_U_P3'.format(id))(P3_out)
    # downsample
    P3_D = MaxPooling2D(strides=(2, 2))(P3_out)
    P4_out = Add()([P3_D, P4_td, P4_in])
    P4_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_D_P4'.format(id))(P4_out)
    P4_D = MaxPooling2D(strides=(2, 2))(P4_out)
    P5_out = Add()([P4_D, P5_td, P5_in])
    P5_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_D_P5'.format(id))(P5_out)
    P5_D = MaxPooling2D(strides=(2, 2))(P5_out)
    P6_out = Add()([P5_D, P6_td, P6_in])
    P6_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_D_P6'.format(id))(P6_out)
    P6_D = MaxPooling2D(strides=(2, 2))(P6_out)
    P7_out = Add()([P6_D, P7_in])
    P7_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_D_P7'.format(id))(P7_out)

    return P3_out, P4_out, P5_out, P6_out, P7_out

def build_regress_head(width, depth, num_anchors=9, detect_quadrangle=False):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': RandomNormal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros'
    }

    inputs = Input(shape=(None, None, width))
    outputs = inputs
    for i in range(depth):
        outputs = Conv2D(
            filters=width,
            activation='relu',
            **options
        )(outputs)

    outputs = Conv2D(num_anchors * 4, **options)(outputs)
    outputs = Reshape((-1, 4))(outputs)

    return Model(inputs=inputs, outputs=outputs, name='box_head')


def build_class_head(width, depth, num_classes=20, num_anchors=9):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
    }

    inputs = Input(shape=(None, None, width))
    outputs = inputs
    for i in range(depth):
        outputs = Conv2D(
            filters=width,
            activation='relu',
            kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
        name='pyramid_classification',
        **options
    )(outputs)
    # (b, num_anchors_this_feature_map, 4)
    outputs = Reshape((-1, num_classes))(outputs)
    outputs = Activation('sigmoid')(outputs)

    return Model(inputs=inputs, outputs=outputs, name='class_head')


def efficientdet(phi=0, num_classes=20, num_anchors=9, weighted_bifpn=False, freeze_bn=False,
                 score_threshold=0.01,
                 detect_quadrangle=False, anchor_parameters=None):
    assert phi in range(7)
    input_size = 512
    input_shape = (input_size, input_size, 3)
    image_input = Input(input_shape)
    w_bifpn = 64
    d_bifpn = 2 + phi
    w_head = w_bifpn
    d_head = 3 + int(phi / 3)
    backbone_cls = inject_tfkeras_modules(EfficientNetB0)
    features = backbone_cls(input_tensor=image_input, freeze_bn=freeze_bn)
    
    for i in range(d_bifpn):
        features = build_BiFPN(features, w_bifpn, i, freeze_bn=freeze_bn)

    regress_head = build_regress_head(w_head, d_head, num_anchors=num_anchors, detect_quadrangle=detect_quadrangle)
    class_head = build_class_head(w_head, d_head, num_classes=num_classes, num_anchors=num_anchors)
    regression = [regress_head(feature) for feature in features]
    regression = Concatenate(axis=1, name='regression')(regression)
    classification = [class_head(feature) for feature in features]
    classification = Concatenate(axis=1, name='classification')(classification)

    # apply predicted regression to anchors
    boxes = RegressBoxes(input_size=input_size, anchor_parameters=anchor_parameters, name='boxes')(regression[..., :4])
    boxes = ClipBoxes(name='clipped_boxes')([image_input, boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = FilterDetections(
        name='filtered_detections',
        score_threshold=score_threshold
    )([boxes, classification])

    prediction_model = Model(inputs=[image_input], outputs=detections, name='efficientdet_p')
    return None, prediction_model
