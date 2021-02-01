import os
import gc
import cv2
import numpy as np
from skimage import transform, io, img_as_float

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from medimodule.utils import Checker
from medimodule.base import BaseModule
from medimodule.Lung.lung_segmentation.utils.postprocessing import _postprocessing

class LungSegmentation(BaseModule):

    def init(self, weight_path):
        """
        Initialize the model with its weight.
        
        Args:
            (string) weight_path : model's weight path
        """
        
        self.model = load_model(weight_path)


    def _preprocessing(self, path):
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
                'Please convert dcm dummies to analyze format.')
            
        Img = img_as_float(io.imread(path))
        Img = transform.resize(Img, (1024,1024))
        Img = np.expand_dims(Img, -1)

        FileName = os.path.split(path)[-1]
        
        Img_mean = Img.mean()
        Img_std  = np.sqrt(((Img**2).mean()) - (Img.mean())**2)

        Img -= Img_mean
        Img /= Img_std

        print("Preprocessing done on {} files...".format(str(FileName)))
        return Img, FileName
    

    
    def predict(self, path):
        """
        Lung segmentation
        
        Args:
            (string) path : image path
            
        Return:
            (numpy ndarray) Lung mask with shape (width, height)
        """
        testData, filenames = self._preprocessing(path)
        testData = np.asarray(testData)
                
        if (len(testData.shape) == 3):
            testData = np.expand_dims(testData, 0)
            
        preds = self.model.predict(testData, batch_size=1)
        preds = preds > 0.5
        mask = _postprocessing(preds)
        print("Test Done !")
        return mask
