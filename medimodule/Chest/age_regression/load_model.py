from tensorflow.keras.applications import densenet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

def build_age_regressor(weight_path):
    img_width, img_height = 512, 512
    model = densenet.DenseNet169(input_shape=(img_width, img_height, 1),
                                    weights=None, include_top=False, pooling='avg')
    x = model.output
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(1, activation='linear')(x)
    model = Model(inputs=model.input, outputs=preds)
    model.load_weights(weight_path)

    return model 
