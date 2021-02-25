import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50


def ViewCls(
    img_height: int = 512,
    img_width: int = 512,
    view_classes: int = 3
) -> Model:
    
    model = ResNet50(input_shape=(img_width, img_height, 1), 
                     weights=None, 
                     include_top=True)
    output = Dense(view_classes, activation='softmax', name='final_layer')(model.output)
    model = Model(inputs=[model.input], outputs=[output])

    return model
