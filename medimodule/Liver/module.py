import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom

from medimodule.utils import Checker
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

        mean_std = [29.311405133024834, 43.38181786843102]
        if Checker.check_input_type_bool(path, 'nii'):
            image = sitk.ReadImage(path)
            self.space = image.GetSpacing()
            image = sitk.GetArrayFromImage(image).astype('float32')
            warnings.warn(
                '.nii is not recommended as an image format '
                'due to be not clear abour horizontal or vertical shape. '
                'Please check the sample in README.md.', UserWarning)

        elif Checker.check_input_type_bool(path, 'dcm'):
            raise ValueError(
                '.dcm is not supported. '
                'Please convert dcm dummies to analyze format.')

        elif Checker.check_input_type_bool(path, 'img') or \
            Checker.check_input_type_bool(path, 'hdr'):
            image = sitk.ReadImage(path)
            self.space = image.GetSpacing()
            image = np.squeeze(sitk.GetArrayFromImage(image).astype('float32')) # (d, w, h)

        elif Checker.check_input_type_bool(path, 'npy'):
            image = np.load(path)
            self.space = [1., 1., 1.]
            warnings.warn(
                '.npy is not recommended as an image format.'
                'Since spacing cannot be identified from .npy, spacing is set as [1., 1., 1.].', 
                UserWarning)

        else:
            input_ext = path.split('.')[-1]
            raise ValueError(
                f'.{input_ext} format is not supported.')

        self.img_shape = image.shape
        d, w, h = self.img_shape

        imageo = image.copy()
        image = zoom(
            image, [self.space[-1]/5., 256./float(w), 256./float(h)], order=1, mode='constant')
        image = np.clip(image, 10, 190)
        image = (image - mean_std[0]) / mean_std[1]
        image = image[np.newaxis,...,np.newaxis] # (1, d, w, h, 1)
        return imageo, image

    def predict(self, path, istogether=False):
        """
        Liver segmentation

        Args:
            (string) path : image path (hdr/img, nii, npy)
            (bool) istogether: with image which was used or not

        Return:
            (numpy ndarray) liver mask with shape (depth, width, height)
        """

        path = os.path.abspath(path)
        imgo, img = self._preprocessing(path)
        mask = np.squeeze(self.model(img).numpy().argmax(axis=-1))
        mask_shape = mask.shape
        mask = zoom(mask, [self.img_shape[0]/mask_shape[0], 
                           self.img_shape[1]/mask_shape[1], 
                           self.img_shape[2]/mask_shape[2]],
                    order=1, mode='constant').astype(np.uint8)
        if istogether:
            return (np.squeeze(imgo), mask)
        return (mask)