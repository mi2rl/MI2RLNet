import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

def EnhanceCls() -> Model:
    model = ResNet50(weights=None, include_top=True, input_shape=(256, 256, 2))
    output = keras.layers.Dense(2, activation='softmax', name='final_layer')(model.output)
    model = keras.models.Model(inputs=[model.input], outputs=[output])

    return model
