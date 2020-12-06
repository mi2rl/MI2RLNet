import os
import sys
import SimpleITK as sitk
import numpy as np
import glob, os
import nibabel as nib
import cv2
import scipy.ndimage as ndimage
import torch

from scipy.ndimage import zoom

from base import BaseModule
from Brain.blackblood_segmentation.models.load_model import build_blackblood_segmentation
from Brain.mri_bet.load_model import build_MRI_BET


class BlackbloodSegmentation(BaseModule):
    def init(self, weight_path):
        """
        Initialize the model with its weight.

        Args:
            (string) weight_path : model's weight path
        """
        self.model = build_blackblood_segmentation(weight_path)



    def _preprocessing(self, path):

        """
        Preprocess the image from the path
        Args:
            (string) path : absolute path of image
        Return:
            (numpy ndarray) image
        """

        windowing_range = [-40., 120.]

        windowing_min = windowing_range[0] - windowing_range[1] // 2
        windowing_max = windowing_range[0] + windowing_range[1] // 2
        image = sitk.ReadImage(path)
        space = image.GetSpacing()
        image = sitk.GetArrayFromImage(image).astype('float32')

        self.img_shape = image.shape
        d, w, h = self.img_shape

        image = ndimage.zoom(image, [.5, .5, .5], order=1, mode='constant', cval=0.)

        image = np.clip(image, windowing_min, windowing_max)
        image = (image - windowing_min) / (windowing_max - windowing_min)
        image = image[np.newaxis, ..., np.newaxis]       # (1, d, w, h, 1)
        return image


    def predict(self, path, istogether=False):
        """
        blackblood segmentation
        (string) path : image path (nii)
        (bool) istogether : with image which was used or not

        """
        path = os.path.abspath(path)
        img = self._preprocessing(path)

        mask = np.squeeze(self.model(img).numpy().argmax(axis=-1))
        mask_shape = mask.shape
        mask = zoom(mask, [self.img_shape[0]/mask_shape[0],
                           self.img_shape[1]/mask_shape[1],
                           self.img_shape[2]/mask_shape[2]],
                    order=1, mode='constant').astype(np.uint8)

        if istogether:
            return (np.squeeze(img), mask)

        # loaded_model = tf.keras.models.load_model(weight_path)

        # model.load_model(weight_path)

        return mask


class MRI_BET(BaseModule):
    def init(self, weight_path):
        """
        Initialize the model with its weight.

        Args:
            (string) weight_path : model's weight path
        """
        self.net = build_MRI_BET(weight_path)


    def _preprocessing(self, path, img_type, min_percent=40, max_percent=98.5, out_min=0, out_max=1):
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


    def _postprocessing(self, data):
        """
        Postprocess the predicted data to reduce FP:wq

        Args:
            (numpy ndarray) 3d data with shape (h, w, d)
        Return:
            (numpy ndarray) 3d data with shape (h, w, d)
        """

        # opening
        kernel = np.ones((5, 5), np.int8)
        data = [cv2.morphologyEx(data[:, :, z], cv2.MORPH_OPEN, kernel) for z in range(data.shape[2])]
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


    def predict(self, path, img_type='T1', save_mask=False, save_stripping=False, thres=0.5):
        """
        Brain tissue segmentation

        Args:
            (string) path : absolute path of data
            (string) img_type : MRI modality type('T1' for T1-weighted MRI, 'MRA' for MR angiography) 
            (bool) save_mask : Boolean type(True for saving binary BET mask. It will be saved in the same path as the input data)
            (bool) save_stripping : Boolean type(True for saving skull-stripped brain image data. It will be saved in the same path as the input data)
            (float) thres : probability threshold to make a mask pixel white (default : 0.5)
        Return:
            (numpy ndarray) 3d brain tissue mask
        """

        data, read_data = self._preprocessing(path, img_type=img_type)
        
        self.net.eval()
        mask3d = np.zeros(np.squeeze(data).shape)
        for z in range(mask3d.shape[2]):
            output = self.net(data[:, :, :, :, z])
            output = torch.sigmoid(output)
            output = output.cpu().detach().numpy().squeeze()
            mask3d[:, :, z] = np.where(output >= thres, 1, 0)

        mask3d = self._postprocessing(mask3d)

        if save_mask is True:
            save_name = path.replace(".nii", "_mask.nii")
            save_ = nib.Nifti1Image(mask3d, read_data.affine, read_data.header)
            nib.save(save_, save_name)

        if save_stripping is True:
            save_name = path.replace(".nii", "_stripping.nii")
            org_img = read_data.get_fdata()
            save_ = nib.Nifti1Image(np.where(mask3d==1,org_img,0), 
                                    read_data.affine, read_data.header)
            nib.save(save_, save_name)

        return mask3d
