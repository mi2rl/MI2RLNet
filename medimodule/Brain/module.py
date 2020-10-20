import os
import SimpleITK as sitk
import numpy as np


## temp ##
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
##########

from base import BaseModule
from preprocessing import Preprocessing
from load_model import build_brain_segmentation

# from medimodule.Brain.brain_segmentation.preprocessing import Preprocessing


class BrainSegmentation(BaseModule):
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

