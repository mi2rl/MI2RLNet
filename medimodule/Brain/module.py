from medimodule.base import BaseModule
from medimodule.Brain.mra_bet.load_model import build_MRA_BET

import glob, os
import nibabel as nib
import numpy as np
import cv2
import scipy.ndimage as ndimage
import torch
import gc

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
        gc.collect()
        
        return mask3d
    
 