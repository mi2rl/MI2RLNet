import tensorflow as tf
from tensorflow.keras.models import Model

from .lr_mark_detection_model.efficientdet import efficientdet

def LRDet() -> Model:
    _, prediction_model = efficientdet(phi=0,
                                       weighted_bifpn=False,
                                       num_classes=2,
                                       score_threshold=0.85)
    
    return prediction_model