from .model import efficientdet
import cv2
import os
import numpy as np
import time
from .utils import preprocess_image
from .utils.anchors import anchors_for_shape
from .utils.draw_boxes import draw_boxes
from .utils.post_process_boxes import post_process_boxes

def build_lr_detection(weight_path : str , gpu_num : str,score_threshold: float):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
    phi = 0
    weighted_bifpn = False
    inner_threshold = score_threshold
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    num_classes = 2
    _, prediction_model = efficientdet(phi=phi,
                                           weighted_bifpn=weighted_bifpn,
                                           num_classes=num_classes,
                                           score_threshold=inner_threshold)
    prediction_model.load_weights(weight_path, by_name=True)
    
    return prediction_model
