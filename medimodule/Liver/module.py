import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom

from medimodule.base import BaseModule
from medimodule.Liver.liver_segmentation.load_model import build_liver_segmentation


class LiverSegmentation(BaseModule):
    def init(self, weight_path):
        """
        Initialize the model with its weight.
        
        Args:
            (string) weight_path : model's weight path
        """

        self.model = build_liver_segmentation(weight_path)

    def _preprocessing(self, path):
        """
        Preprocess the image from the path

        Args:
            (string) path : absolute path of image
        Return:
            (numpy ndarray) image
        """
        
        '''
        TODO : check image format
            - hdr : pass
            - nii : ?
            - dcm : ?
        '''

        mean_std = [29.311405133024834, 43.38181786843102]
        image = sitk.ReadImage(path)
        space = image.GetSpacing()
        image = np.squeeze(sitk.GetArrayFromImage(image).astype('float32')) # (d, w, h)
        d, w, h = image.shape
        image = zoom(image, [space[-1]/5., 256./float(w), 256./float(h)], order=1, mode='constant')
        image = np.clip(image, 10, 190)
        image = (image - mean_std[0]) / mean_std[1]
        image = image[np.newaxis,...,np.newaxis] # (1, d, w, h, 1)
        return image

    def predict(self, path, istogether=False):
        """
        Liver segmentation

        Args:
            (string) path : hdr path
            (bool) istogether: with image which was used or not

        Return:
            (numpy ndarray) liver mask with shape (depth, width, height)
        """

        img = self._preprocessing(path)
        mask = np.squeeze(self.model(img).numpy().argmax(axis=-1))
        if istogether:
            return (np.squeeze(img), mask)
        return (mask)