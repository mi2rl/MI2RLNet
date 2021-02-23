from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50

def ViewClassifier(weight_path):
    img_width, img_height = 512, 512
    view_classes = 3

    model = ResNet50(input_shape=(img_width, img_height, 1), 
                        weights=None, include_top=True)
    output = keras.layers.Dense(view_classes, activation='softmax', name='final_layer')(model.output)
    model = keras.models.Model(inputs=[model.input], outputs=[output])
    model.load_weights(weight_path)

    return model
