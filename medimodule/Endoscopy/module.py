import cv2
import torch
import numpy as np
import sys
sys.path.append('../')

from base import BaseModule
from Endoscopy.polyp_segmentation.load_model import build_polyp_segmentation
from utils import Checker

class PolypSegmentation(BaseModule):
    def init(self, weight_path):
        """
        Initialize the model with its weight.
        
        Args:
            (string) weight_path : model's weight path
        """

        self.model = build_polyp_segmentation(weight_path)

    def _preprocessing(self, path):
        """
        Preprocess the image from the path
        Args:
            (string) path : absolute path of image
        Return:
            (numpy ndarray) image
        """
        
        '''
        TODO : ?
        '''
        Checker.check_input_type(path, 'png')
        image = cv2.imread(path)
        image = cv2.resize(image, dsize=(512, 512))
        image = image/255.0
        image = image.transpose((2, 0, 1)).astype(np.float32)
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        return image

    def predict(self, path, thresh=0.7):
        """
        Liver segmentation
        Args:
            (string) path : image path
            (bool) thresh : the value of pixel which you will start to use as polyp Lesions(0 ~ 1). 
                            if it is 0.7, the pixel which has value under 0.7 won't be used as Lesion pixel.
        Return:
            (numpy ndarray) polyp mask with shape (width, height)
        """
        fn_thresh = lambda x, thresh :  1.0 * (x > thresh)
        
        img = self._preprocessing(path)
        mask = self.model(img) 
        mask = fn_thresh(mask, thresh)
        mask = mask.numpy()
        return mask
