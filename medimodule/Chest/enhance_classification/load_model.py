from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50

def build_enhanceCT_classifier(weight_path, in_ch):
    w, h, c = 256, 256, in_ch
    n_classes = 2

    model = ResNet50(weights=None, include_top=True, input_shape=(w, h, c)
    output = keras.layers.Dense(n_classes, activation='softmax', name='final_layer')(model.output)
    model = keras.models.Model(inputs=[model.input], outputs=[output])
    model.load_weights(weight_path)

    return model
