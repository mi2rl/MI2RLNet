import os
import sys
import cv2
import glob
import warnings
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import scipy.ndimage as ndimage
from typing import Tuple, Optional

import torch

from medimodule.utils import Checker
from medimodule.base import BaseModule
from medimodule.Brain.models import MRIBET
from medimodule.Brain.models import BBBSeg


class BlackbloodSegmentation(BaseModule):
    def __init__(self, weight_path: str = None):
        """
        Initialize the model with its weight.
        Args:
            (string) weight_path : model's weight path
        """
        self.model = BBBSeg()
        if weight_path is not None:
            self.model.load_weights(weight_path)

    def _preprocessing(self, path: str) -> np.array:
        """
        Preprocess the image from the path
        Args:
            (string) path : absolute path of image
        Return:
            (numpy ndarray) image shape (1, h, w, d, 1)
        """
        if Checker.check_input_type_bool(path, 'nii'):
            image = sitk.ReadImage(path)
            self.space = image.GetSpacing()
            image = sitk.GetArrayFromImage(image).astype('float32')

        elif Checker.check_input_type_bool(path, 'npy'):
            image = np.load(path)
            self.space = [1., 1., 1.]
            warnings.warn(
                '.npy is not recommended as an image format.'
                'Since spacing cannot be identified from .npy, spacing is set as [1., 1., 1.].', UserWarning)

        elif Checker.check_input_type_bool(path, 'dcm'):
            raise ValueError(
                '.dcm is not supported.'
                'Please convert dcm dummies to nii format.')

        else:
            input_ext = path.split('.')[-1]
            raise ValueError(
                f'.{input_ext} format is not supported.')

        self.img_shape = image.shape

        # normalize
        windowing_range = [-40., 120.]
        windowing_min = windowing_range[0] - windowing_range[1] // 2
        windowing_max = windowing_range[0] + windowing_range[1] // 2
        image = ndimage.zoom(image, [.5, .5, .5], order=1, mode='constant')
        image = np.clip(image, windowing_min, windowing_max)
        image = (image - windowing_min) / (windowing_max - windowing_min)
        image = image[np.newaxis, ..., np.newaxis]
        return image

    def predict(
        self, 
        path: str, 
        save_path: Optional[str] = None
    ) -> Tuple[np.array, np.array]:
        """
        blackblood segmentation
        Args:
            (string) path : image path (nii)
            (bool) istogether : with image which was used or not
        Return:
            (numpy ndarray) blackblood mask with shape
        """
        path = os.path.abspath(path)
        image = self._preprocessing(path)
        mask = np.squeeze(self.model(image).numpy().argmax(axis=-1))
        mask_shape = mask.shape
        mask = ndimage.zoom(mask, [self.img_shape[0]/ mask_shape[0],
                                   self.img_shape[1]/ mask_shape[1],
                                   self.img_shape[2]/ mask_shape[2]],
                            order=1, mode='constant')
        mask = mask.astype(np.uint8)

        if save_path:
            temp2 = np.swapaxes(mask, 1, 2)
            temp2 = np.swapaxes(temp2, 0, 1)
            temp2 = np.swapaxes(temp2, 1, 2)
            mask_pair = nib.Nifti1Pair(temp2, np.diag([-self.space[0],
                                                       self.space[1],
                                                       self.space[2], 1]))
            nib.save(mask_pair, save_path)

        return (np.squeeze(image), mask)


class MRI_BET(BaseModule):
    def __init__(self, weight_path: str = None):
        """
        Initialize the model with its weight.

        Args:
            (string) weight_path : model's weight path
        """
        self.model = MRIBET().cuda()
        if weight_path is not None:
            weight = torch.load(weight_path, map_location="cuda:0")
            self.model.load_state_dict(weight["net"])

    def _preprocessing(
        self, 
        path: str, 
        img_type: str, 
        min_percent: float = 40., 
        max_percent: float = 98.5, 
        out_min: int = 0, 
        out_max: int = 1) -> torch.Tensor:
        """
        Preprocess the image from the path

        Args:
            (string) path : absolute path of data
            (string) img_type : MRI modality type for applying different preprocessing according to MRI modality
            (float) min_percent : Min percentile to compute, which must be between 0 and 100 inclusive (default : 40)
            (float) max_percent : Max percentile to compute, which must be between 0 and 100 inclusive (default : 98.5)
            (integer) out_min : minimum value of output (default : 0)
            (integer) out_max : maximum value of output (default : 1)
        Return:
            (numpy ndarray) data with shape (1, h, w, d, 1)

        """

        read_data = nib.load(path)
        data = read_data.get_fdata().astype(np.float32)

        if img_type == 'T1':
            pass
        elif img_type == 'MRA':
            # windowing
            w_min = np.percentile(data, min_percent)
            w_max = np.percentile(data, max_percent)
            width = w_max - w_min + 1
            center = w_min + width / 2

            data = ((data - center) / width + 0.5) * (out_max - out_min)
            data = np.piecewise(data, [data <= out_min, data >= out_max],
                                [out_min, out_max, lambda data: data])

        # ToTensor, unsqueezing
        data = torch.from_numpy(data)[np.newaxis, np.newaxis, ...]
        data = data.cuda()
        return data, read_data


    def _postprocessing(self, data: np.array) -> np.array:
        """
        Postprocess the predicted data to reduce FP:wq

        Args:
            (numpy ndarray) 3d data with shape (h, w, d)
        Return:
            (numpy ndarray) 3d data with shape (h, w, d)
        """

        # opening
        kernel = np.ones((5, 5), np.int8)
        data = [cv2.morphologyEx(data[:, :, z], cv2.MORPH_OPEN, kernel) 
                for z in range(data.shape[2])]
        data = np.transpose(np.asarray(data), (1, 2, 0))

        # FP reduction using ccl
        img_labels, num_labels = ndimage.label(data)
        sizes = ndimage.sum(data, img_labels, range(num_labels + 1))
        remove_cluster = sizes < np.max(sizes)
        remove_pixel = remove_cluster[img_labels]
        data[remove_pixel] = 0
        data[data > 0] = 1

        # fill hole
        data = ndimage.binary_fill_holes(data).astype(np.float32)
        
        return data


    def predict(
        self, 
        path: str, 
        img_type: str = 'T1', 
        save_path: Optional[str] = None, 
        thres: float = 0.5
    ) -> Tuple[np.array, np.array]:
        """
        Brain tissue segmentation

        Args:
            (string) path : absolute path of data
            (string) img_type : MRI modality type('T1' for T1-weighted MRI, 'MRA' for MR angiography) 
            (string) save_path : If save_path is set, the mask and the skull-stripped image will be saved.
            (float) thres : probability threshold to make a mask pixel white (default : 0.5)
        Return:
            (numpy ndarray) 3d brain tissue mask
        """

        data, read_data = self._preprocessing(path, img_type=img_type)
        
        self.model.eval()
        mask = np.zeros(np.squeeze(data).shape)
        for z in range(mask.shape[2]):
            output = self.model(data[:, :, :, :, z])
            output = torch.sigmoid(output)
            output = output.cpu().detach().numpy().squeeze()
            mask[:, :, z] = np.where(output >= thres, 1, 0)

        mask = self._postprocessing(mask)
        mask = mask.astype(np.uint8)

        if save_path:
            mask_save = nib.Nifti1Image(mask, read_data.affine, read_data.header)
            nib.save(mask_save, save_path)
            
            org_img = read_data.get_fdata()
            img_strip_save = nib.Nifti1Image(
                np.where(mask == 1, org_img, 0), 
                read_data.affine, read_data.header)
            nib.save(img_strip_save, path.replace(".nii", "_stripped.nii"))

        return (np.squeeze(data), mask)