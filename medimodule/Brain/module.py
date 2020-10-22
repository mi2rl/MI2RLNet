import os
import sys
import SimpleITK as sitk
import numpy as np
import glob, os
import nibabel as nib
import cv2
import scipy.ndimage as ndimage
import torch

from base import BaseModule

from medimodule.Brain.blackblood_segmentation.preprocessing import Preprocessing
from medimodule.Brain.blackblood_segmentation.load_model import build_blackblood_segmentation

from medimodule.base import BaseModule
from medimodule.Brain.mra_bet.load_model import build_MRA_BET


class BlackbloodSegmentation(BaseModule):
    def init(self, weight_path):
        """
        Initialize the model with its weight.

        Args:
            (string) weight_path : model's weight path
        """
        self.model = build_brain_segmentation(weight_path)


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
            - hdr : ?
            - nii : pass
            - dcm : ?
        '''

        prep = Preprocessing(args,
                             rotation_range=[0., 0., 0.],
                             voi_dict=None)

        cases = [c for c in os.listdir(path) if 'T1.nii' in c]
        cases = sorted([c.split('.')[0][:-3] for c in cases])

        for data in cases:
            img_num = data[:20]
            img_noise = data[-3:]
            img = sitk.ReadImage(os.path.join(path, img_num + '_T1_{}.nii'.format(img_noise)))

            img = prep._windowing(img)
            img = prep._standardize(img)
            img = img[np.newaxis, ..., np.newaxis]

            yield img
        return img

    def predict(self, path, istogether=False):
        img = self._preprocessing(path)
        mask = np.squeeze(self.model(img).numpy().argmax(axis=-1))
        if istogether:
            return (np.sqqueeze(img), mask)
        return mask


class MRA_BET(BaseModule):
    def init(self, weight_path, gpu_num):
        """
        Initialize the model with its weight.
        
        Args:
            (string) weight_path : model's weight path
            (scalar) gpu_num : select GPU number
        """

        self.net, self.device = build_MRA_BET(weight_path, gpu_num)

        
    def _preprocessing(self, path, min_percent=40, max_percent=98.5, out_min=0, out_max=1):
        """
        Preprocess the image from the path

        Args:
            (string) path : absolute path of data
            (float) min_percent : Min percentile to compute, which must be between 0 and 100 inclusive (default : 40)
            (float) max_percent : Max percentile to compute, which must be between 0 and 100 inclusive (default : 98.5)
            (integer) out_min : minimum value of output (default : 0)
            (integer) out_max : maximum value of output (default : 1)
        Return:
            (numpy ndarray) data with shape (1, h, w, d, 1)
        """ 

        data = nib.load(path).get_fdata().astype(np.float32)

        # normalizing
        w_min = np.percentile(data, min_percent)
        w_max = np.percentile(data, max_percent)
        width = w_max - w_min + 1
        center = w_min + width / 2
        data = ((data-center)/width+0.5)*(out_max-out_min)
        data = np.piecewise(data, [data <= out_min,data >= out_max],
                            [out_min, out_max, lambda data: data])
                
        # ToTensor, unsqueezing
        data = torch.from_numpy(data)[np.newaxis,np.newaxis,...]
        data = data.to(self.device, dtype=torch.float32)
        return data

    
    def _postprocessing(self, data):
        """
        Postprocess the predicted data to reduce FP:wq

        Args:
            (numpy ndarray) 3d data with shape (h, w, d)
        Return:
            (numpy ndarray) 3d data with shape (h, w, d)
        """ 
        
        # opening
        kernel = np.ones((5,5), np.int8)
        data = [cv2.morphologyEx(data[:,:,z],cv2.MORPH_OPEN,kernel) for z in range(data.shape[2])]
        data = np.transpose(np.asarray(data),(1,2,0))
        
        # FP reduction using ccl
        img_labels, num_labels = ndimage.label(data)
        sizes = ndimage.sum(data, img_labels, range(num_labels+1))
        remove_cluster = sizes < np.max(sizes)
        remove_pixel = remove_cluster[img_labels]
        data[remove_pixel] = 0
        data[data > 0] = 1
        
        # fill hole
        data = ndimage.binary_fill_holes(data).astype(np.float32)
        
        return data
    
    def predict(self, path, thres=0.5):
        """
        Brain tissue segmentation

        Args:
            (string) path : absolute path of data
            (float) thres : probability threshold to make a mask pixel white (default : 0.5)
        Return:
            (numpy ndarray) 3d brain tissue mask 
        """ 

        data = self._preprocessing(path)
        
        self.net.eval()
        mask3d = np.zeros(np.squeeze(data).shape)
        for z in range(mask3d.shape[2]):
            output = self.net(data[:,:,:,:,z])
            output = torch.sigmoid(output)
            output = output.cpu().detach().numpy()
            mask3d[:,:,z] = 1*(np.squeeze(output) >= thres)
        
        mask3d = self._postprocessing(mask3d)
        
        return mask3d
    
