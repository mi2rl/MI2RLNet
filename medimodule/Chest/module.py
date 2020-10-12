import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import SimpleITK as sitk

from ..base import BaseModule
from .age_regression.load_model import build_age_regressor
from .viewpoint_classification.load_model import build_view_classifier
from .enhance_classification.load_model import build_enhanceCT_classifier


class AgeRegressor(BaseModule):
    def init(self, weight_path):
        self.model = build_age_regressor(weight_path)

    def _preprocessing(self, path):
        """
        Image preprocessing for classifying Viewpoint

        Args:
            (string) path : dicom path
        Return:
            (numpy ndarray) img 
        """
        img = sitk.GetArrayFromImage(sitk.ReadImage(path))
        img = img[0]

        img = cv2.resize(img, (512,512), interpolation=cv2.INTER_LINEAR)

        np_img = img.astype(np.float32)
        np_img -= np.min(np_img)
        np_img /= np.percentile(np_img, 99)

        np_img[np_img>1] = 1
        np_img *= (2**8-1)
        np_img = np_img.astype(np.uint8)
        
        # convert shape [b, h, w, c]
        np_img = np.expand_dims(np.expand_dims(np_img, 0), -1)
        return np_img
    
    def predict(self, path):
        """
        Age prediction 

        Args:
            (numpy ndarray) img : numpy array which have (H, W, C) shape.
        Return:
            (int) age : age prediction result (Month)
        """
        img = self._preprocessing(path)
        print(img.shape)
        assert (img.shape[1] == 512) & (img.shape[2] == 512), "The size of image must be (batch, 512, 512, 1)"
        # in training pahse, we normalize value by divide 1200.
        # so if we want to get real value(age), have to multiply 1200.
        return (self.model.predict(img))

class ViewpointClassifier(BaseModule):
    """ Classify PA / Lateral / Others View """

    def init(self, weight_path):
        self.model = build_view_classifier(weight_path)

    def _preprocessing(self, path):
        """
        Image preprocessing for classifying Viewpoint

        Args:
            (string) path : dicom path
        Return:
            (numpy ndarray) img 
        """
        img = sitk.GetArrayFromImage(sitk.ReadImage(path))
        img = img[0]

        img = cv2.resize(img, (512,512), interpolation=cv2.INTER_LINEAR)

        np_img = img.astype(np.float32)
        np_img -= np.min(np_img)
        np_img /= np.percentile(np_img, 99)

        np_img[np_img>1] = 1
        np_img *= (2**8-1)
        np_img = np_img.astype(np.uint8)
        
        # convert shape [b, h, w, c]
        np_img = np.expand_dims(np.expand_dims(np_img, 0), -1)

        return np_img
        
    def predict(self, path):
        """
        View Classification

        Args:
            (numpy ndarray) img : numpy array which have (H, W, C) shape.
        Return:
            (int) age : age prediction result (Month)
        """
        img = self._preprocessing(path)
        pred = self.model.predict(img)
        out = np.argmax(pred)
        labels = ['PA', 'Lateral', 'Others']
        return labels[out]

import pydicom
from pydicom import dcmread
from pydicom.pixel_data_handlers import gdcm_handler
from numpy import newaxis

class EnhanceCTClassifier(BaseModule):
    """ Classify Enhanced CT vs Non-Enhanced CT """

    def init(self, weight_path, in_ch=2):
        self.in_ch = in_ch
        self.model = build_enhanceCT_classifier(weight_path, self.in_ch)

    def _preprocessing(self, path):
        """
        Image preprocessing for classifying Enhanced CT vs. Non-Enhanced CT

        Args:
            (numpy ndarray) img : numpy array which have (H, W, C) shape.
        Return:
            (numpy ndarray) results : preprocessed image
        """
        # image size settings
        h, w, c = 256, 256, 2
        
        ds = dcmread(path)
        
        # for JPEG Lossless, Nonhierarchical, First- Order Prediction
        if ds.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.70':
            ds.decompress()
        
        img = ds.pixel_array
        try:
            intercept = ds.RescaleIntercept # if not CT, it will raise error
            tmp_img = img.astype('float32')
            tmp_img2 = np.copy(tmp_img)
            
            window_width = 2048.
            window_level = 1024.

            lower = (window_level-window_width)
            upper = (window_level+window_width)

            tmp_img -= (lower - intercept)
            tmp_img /= (upper + 1024)
            tmp_img[tmp_img < 0] = 0.
            tmp_img[tmp_img > 1] = 1.
            tmp_img = cv2.resize(tmp_img, (h, w), interpolation = cv2.INTER_AREA)
            tmp_img = tmp_img[:, :, newaxis]

            tmp_img2[tmp_img2 == -2000] = 0.
            tmp_img2 -= (-1024. - intercept)
            tmp_img2 /= 4096
            tmp_img2 = cv2.resize(tmp_img2, (height, width), interpolation = cv2.INTER_AREA)
            tmp_img2 = tmp_img2[:, :, newaxis]

            results = np.concatenate((tmp_img, tmp_img2), axis=2)
        except:
            print('please check your image modality, you have to input CT')
            raise
        return results

    def predict(self, path):
        img = self._preprocessing(path)
        pred = self.model.predict(img)
        out = np.argmax(pred)
        labels = ['Non-Enhanced', 'Enhanced']
        return labels[out]



