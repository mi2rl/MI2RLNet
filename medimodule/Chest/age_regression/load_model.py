from keras.applications import densenet
from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras import regularizers
from keras import backend as K

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
