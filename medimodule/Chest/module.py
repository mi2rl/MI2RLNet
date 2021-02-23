import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import SimpleITK as sitk

from medimodule.utils import Checker
from medimodule.base import BaseModule
from medimodule.Chest.models import LRmarkDetection
from medimodule.Chest.models import ViewClassifier
from medimodule.Chest.models import EnhanceClassification

class ViewpointClassifier(BaseModule):
    """ Classify PA / Lateral / Others View """

    def __init__(self, weight_path: str = None):
        self.model = ViewClassifier(weight_path)

    def _preprocessing(self, path: str) -> np.array:
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
        
    def predict(self, path: str) -> int:
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

    def __init__(self, weight_path: str = None, in_ch: int = 2):
        self.in_ch = in_ch
        self.model = ViewClassifier(weight_path, self.in_ch)

    def _preprocessing(self, path: str) -> np.array:
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

    def predict(self, path: str) -> str:
        img = self._preprocessing(path)
        pred = self.model.predict(img)
        out = np.argmax(pred)
        labels = ['Non-Enhanced', 'Enhanced']
        return labels[out]

from medimodule.Chest.models.utils.anchors import anchors_for_shape
from medimodule.Chest.models.utils.post_process_boxes import post_process_boxes

class ChestLRmarkDetection(BaseModule):
    def __init__(self, weight_path: str = None):
        """
        Initialize the model with its weight.
        
        Args:
            (string) weight_path : model's weight path
        """

        self.model = LRmarkDetection(weight_path)
        
    def _preprocessing(self, path: str) -> tuple:
        """
        Preprocess the image from the path
            - png : mean/SD scaling
        Args:
            (string) path : absolute path of image
        Return: tuple
            (numpy ndarray) new_image : scaled image
            (numpy ndarray) src_image : origin_image
            (float) scale, offset_h, offset_w , image_size , h,w : scaled image informations
        """
        
        '''
        Post Processing about predict bounding box
        
            - scaling Bounding Box using offset
            
        '''
        
        if '.png' in path or '.jpg' in path or '.bmp' in path:
            
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            src_image = image.copy()
            image = image[:, :, ::-1]
            h, w = image.shape[:2]
            
            image_height, image_width = image.shape[:2]
            image_size = 512 # b0 baseline
            
            if image_height > image_width:
                scale = image_size / image_height
                resized_height = image_size
                resized_width = int(image_width * scale)
            else:
                scale = image_size / image_width
                resized_height = int(image_height * scale)
                resized_width = image_size

            image = cv2.resize(image, (resized_width, resized_height))
            new_image = np.ones((image_size, image_size, 3), dtype=np.float32) * 128.
            offset_h = (image_size - resized_height) // 2
            offset_w = (image_size - resized_width) // 2
            new_image[offset_h:offset_h + resized_height, offset_w:offset_w + resized_width] = image.astype(np.float32)
            new_image /= 255.

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            
            for i in range(3):
                new_image[..., i] -= mean[i]
                new_image[..., i] /= std[i]
        
        return (new_image, scale, offset_h, offset_w , image_size , src_image, h,w)

    def predict(self, path:str) -> np.array:
        """
        L,R Detection 
        Args:
            (string) path : 8-bit png path
            
        Return:
            (numpy ndarray) L,R detection image (W,H,C)
        """
        classes = ['left','right']
        num_classes = len(classes)
        score_threshold = 0.85
        (image, scale, offset_h, offset_w ,image_size ,src_image,h,w) = self._preprocessing(path)
        
        anchors = anchors_for_shape((image_size, image_size))
        predict = np.zeros((3,6),dtype=np.float32)

        boxes, scores, labels = self.model.predict([np.expand_dims(image, axis=0),
                                                                   np.expand_dims(anchors, axis=0)])
        
        boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
        boxes = post_process_boxes(boxes=boxes,
                               scale=scale,
                               offset_h=offset_h,
                               offset_w=offset_w,
                               height=h,
                               width=w)
    
        indices = np.where(scores[:] > score_threshold)[0]
        boxes = boxes[indices]
        labels = labels[indices]
        
        for b, l, s in zip(boxes, labels, scores):
            class_id = int(l)
            class_name = classes[class_id]

            xmin, ymin, xmax, ymax = list(map(int, b))
            score = '{:.4f}'.format(s)
            color = [0,0,255]
            label = '-'.join([class_name, score])

            ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), color, 3)
            cv2.rectangle(src_image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
            cv2.putText(src_image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


        return (src_image)


