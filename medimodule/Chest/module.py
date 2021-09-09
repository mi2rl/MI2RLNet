import os
import sys
import cv2
import json
import pydicom
import warnings
import numpy as np
import SimpleITK as sitk
from typing import Tuple, Optional, List, Dict

from medimodule.utils import Checker
from medimodule.base import BaseModule
from medimodule.Chest.models import LungSeg
from medimodule.Chest.models import LRDet
from medimodule.Chest.models import ViewCls
from medimodule.Chest.models import EnhanceCls

from medimodule.Chest.models.lr_mark_detection_model.anchors import anchors_for_shape


class ViewpointClassifier(BaseModule):
    def __init__(self, weight_path: Optional[str] = None):
        """
        Classifying PA / Lateral / Others
        
        Args:
            (string) weight_path : pretrained weight path (optional)
        """

        self.model = ViewCls()
        if weight_path:
            self.model.load_weights(weight_path)

        self.labels = ['PA', 'Lateral', 'Others']

    def _preprocessing(self, path: str) -> Tuple[np.array, np.array]:
        """
        Args:
            (string) path : dicom path
        Return:
            (numpy ndarray) imgo : original image
            (numpy ndarray) img  : preprocessed image
        """
        
        imgo = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(path)))
        img = cv2.resize(imgo, (512, 512), interpolation=cv2.INTER_LINEAR)

        img = img.astype(np.float32)
        img -= np.min(img)
        img /= np.percentile(img, 99)

        img[img > 1] = 1
        img *= (2**8-1)
        img = img.astype(np.uint8)
        
        img = img[None,...,None]

        return imgo, img
        
    def predict(
        self, 
        path: str,
        save_path: Optional[str] = None
    ) -> Tuple[np.array, str]:
        """
        Args:
            (string) path : image path
            (string) save_path : save path
        Return:
            (Tuple) (imgo, result)
            (numpy ndarray) imgo : original image
            (string) result : PA / Lateral / Others
        """

        path = os.path.abspath(path)
        imgo, img = self._preprocessing(path)
        result = self.model.predict(img)
        result = np.argmax(result)
        result = self.labels[result]
        
        if save_path:
            if ".txt" in save_path:
                with open(save_path, "w") as f:
                    f.write(result)
            else:
                warnings.warn(
                    "ONLY txt format is allowed in this module."
                    "If you want other format, use a custom code.",
                    UserWarning)

        return (imgo, result)


class EnhanceCTClassifier(BaseModule):
    def __init__(self, weight_path: Optional[str] = None):
        """
        Classify Enhanced CT vs Non-Enhanced CT
        Args:
            (string) weight_path : pretrained weight path (optional)
        """

        self.model = EnhanceCls()
        if weight_path:
            self.model.load_weights(weight_path)

        self.labels = ['Non-Enhanced', 'Enhanced']

    def _preprocessing(self, path: str) -> Tuple[np.array, np.array]:
        """
        Args:
            (string) path : dicom path
        Return:
            (numpy ndarray) imgo : original image
            (numpy ndarray) img  : preprocessed image
        """
        # image size settings
        
        ds = pydicom.dcmread(path)
        
        # for JPEG Lossless, Nonhierarchical, First- Order Prediction
        if ds.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.70':
            ds.decompress()
        
        imgo = ds.pixel_array
        try:
            intercept = ds.RescaleIntercept # if not CT, it will raise error
            tmp_img = imgo.astype('float32')
            tmp_img2 = np.copy(tmp_img)
            
            window_width = 450.
            window_level = 550.

            lower = (window_level-window_width)
            upper = (window_level+window_width)

            tmp_img -= (lower - intercept)
            tmp_img /= (upper + 1024)
            tmp_img[tmp_img < 0] = 0.
            tmp_img[tmp_img > 1] = 1.
            tmp_img = cv2.resize(tmp_img, (256, 256), interpolation=cv2.INTER_AREA)
            tmp_img = tmp_img[:,:,None]

            tmp_img2[tmp_img2 == -2000] = 0.
            tmp_img2 -= -1024. - intercept
            tmp_img2 /= 4096
            tmp_img2 = cv2.resize(tmp_img2, (256, 256), interpolation=cv2.INTER_AREA)
            tmp_img2 = tmp_img2[:,:,None]

            img = np.concatenate((tmp_img, tmp_img2), axis=-1)
            img = np.expand_dims(img, axis=0)

        except:
            print('please check your image modality, you have to input CT')
            raise

        return imgo, img

    def predict(
        self, 
        path: str,
        save_path: Optional[str] = None
    ) -> Tuple[np.array, str]:

        path = os.path.abspath(path)
        imgo, img = self._preprocessing(path)
        result = self.model.predict(img)
        result = np.argmax(result)
        result = self.labels[result]

        if save_path:
            if ".txt" in save_path:
                with open(save_path, "w") as f:
                    f.write(result)
            else:
                warnings.warn(
                    "ONLY txt format is allowed in this module."
                    "If you want other format, use a custom code.",
                    UserWarning)
        
        return (imgo, result)


class LRmarkDetection(BaseModule):
    def __init__(self, weight_path: Optional[str] = None):
        """
        Initialize the model with its weight.
        
        Args:
            (string) weight_path : model's weight path
        """

        self.model = LRDet()
        if weight_path:
            self.model.load_weights(weight_path, by_name=True)

        self.labels = ['left', 'right']
        
    def _preprocessing(
        self, path: str
    ) -> Tuple[np.array, float, int, int, int, np.array, int, int]:

        """
        Preprocess the image from the path
            - png : mean/SD scaling
        Args:
            (string) path : image path
        Return: tuple
            (numpy ndarray) new_image : scaled image
            (float) scale, offset_h, offset_w, image_size, h, w : scaled image informations
            (numpy ndarray) imgo : original image
        """
        
        '''
        Post Processing about predict bounding box
        
            - scaling Bounding Box using offset
            
        '''
        
        if '.png' in path or '.jpg' in path or '.bmp' in path:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imgo = image.copy()
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
            new_image[offset_h:offset_h+resized_height, 
                      offset_w:offset_w+resized_width] = image.astype(np.float32)
            new_image /= 255.

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            
            for i in range(3):
                new_image[..., i] -= mean[i]
                new_image[..., i] /= std[i]
        
        return (new_image, scale, offset_h, offset_w, image_size, imgo, h, w)

    def _postprocessing(
        self, 
        boxes: np.array, 
        scale: np.array, 
        offset_h: int, 
        offset_w: int, 
        height: int, 
        width: int
    ) -> np.array:

        boxes[:, [0, 2]] = boxes[:, [0, 2]] - offset_w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] - offset_h
        boxes /= scale
        boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)

        return boxes

    def predict(
        self, 
        path: str,
        save_path: Optional[str] = None,
        score_threshold: float = 0.85
    ) -> Tuple[np.array, Dict]:

        """
        L,R Detection 
        Args:
            (string) path : 8-bit png path
            
        Return:
            (numpy ndarray) L,R detection image (W,H,C)
        """
        
        path = os.path.abspath(path)
        img, scale, offset_h, offset_w, image_size, imgo, h, w = self._preprocessing(path)
        anchors = anchors_for_shape((image_size, image_size))

        boxes, scores, labels = self.model.predict(
            [np.expand_dims(img, axis=0), np.expand_dims(anchors, axis=0)])
        
        boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
        boxes = self._postprocessing(
            boxes=boxes,
            scale=scale,
            offset_h=offset_h,
            offset_w=offset_w,
            height=h,
            width=w)
    
        indices = np.where(scores[:] > score_threshold)[0]
        boxes = boxes[indices]
        labels = labels[indices]
        results = {'left': [], 'right': []}
        for b, l, s in zip(boxes, labels, scores):
            class_name = self.labels[int(l)]
            xmin, ymin, xmax, ymax = list(map(int, b))
            results[class_name].append([xmin, ymin, xmax, ymax, float(s)])
        
        if save_path:
            if '.json' in save_path:
                json.dump(results, open(save_path, 'w'), indent='\t')
            else:
                warnings.warn(
                    "ONLY json format is allowed in this module."
                    "If you want other format, use a custom code.",
                    UserWarning)

        return (imgo, results)


class LungSegmentation(BaseModule):
    def __init__(self, weight_path: Optional[str] = None):
        """
        Initialize the model with its weight.
        
        Args:
            (string) weight_path : model's weight path
        """
        
        # TODO: separate model and weights
        self.model = LungSeg()
        if weight_path:
            self.model.load_weights(weight_path)

    def _preprocessing(self, path: str) -> Tuple[np.array, np.array]:
        """
        Preprocess the image from the path
        
        Args:
            (string) path : absolute path of image
        Return:
            (numpy ndarray) image
        """
        
        if Checker.check_input_type_bool(path, 'dcm'):
            raise ValueError(
                '.dcm is not supported. '
                'Please convert dcm format to image format, such as png or jpg.')
            
        imageo = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        image = cv2.resize(imageo, (1024, 1024), interpolation=cv2.INTER_AREA)
        if len(image.shape) == 2:
            image = image[None,...,None]

        image -= image.mean()
        image /= image.std()

        return imageo, image

    def _postprocessing(self, mask: np.array) -> np.array:
        mask = mask[0].astype(np.uint8)
        mask *= 255
        kernel = np.ones((5, 5), np.uint8)

        erosion = cv2.erode(mask, kernel, iterations=1)
        dilation = cv2.dilate(erosion, kernel, iterations=1)
        _, thresh = cv2.threshold(dilation, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        areas = []
        for cnt in contours:
            areas.append(cv2.contourArea(cnt))

        if len(areas) == 0:
            return np.zeros_like(mask)

        areas = np.array(areas)
        maxindex = np.argmax(areas)
        areas[maxindex] = 0
        secondmaxindex = np.argmax(areas)

        for i, cnt in enumerate(contours):
            if i != maxindex and i != secondmaxindex:
                cv2.drawContours(dilation, contours, i, color=(0, 0, 0), thickness=-1)

        erosion = cv2.erode(dilation, kernel, iterations=1)
        img_post = cv2.dilate(erosion, kernel, iterations=1)

        if len(img_post.shape) < 3:
            img_post = np.expand_dims(img_post, axis=-1)

        return img_post
    
    def predict(
        self, 
        path: str,
        save_path: Optional[str] = None
    ) -> Tuple[np.array, np.array]:
        """
        Lung segmentation
        
        Args:
            (string) path : image path
            
        Return:
            (numpy ndarray) Lung mask with shape (width, height)
        """

        path = os.path.abspath(path)
        imgo, img = self._preprocessing(path)
            
        mask = self.model.predict(img, batch_size=1)
        mask = mask > 0.5
        mask = self._postprocessing(mask)
        if save_path:
            cv2.imwrite(save_path, mask)

        return (imgo, mask)
