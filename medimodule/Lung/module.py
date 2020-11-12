import os
import gc
import cv2
import numpy as np
from skimage import transform, io, img_as_float32


from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from base import BaseModule
from .lung_segmentation.utils.postprocessing import _postprocessing

class LungSegmentation(BaseModule):

    def init(self, weight_path):
        """
        Initialize the model with its weight.
        
        Args:
            (string) weight_path : model's weight path
        """
        
        self.model = load_model(weight_path)


    def _preprocessing(self, ImgPath):
        """
        Preprocess the image from the path
        
        Args:
            (string) path : file path of image
        Return:
            (numpy ndarray) image
        """
        
        '''
        TODO : check image format
            - png : pass
        '''
        
        Imgs = []
        FileNames = []

        Imgs_mean = 0.
        Imgs_meansq = 0.

        for i, filename in enumerate(os.listdir(ImgPath)):

            if (os.path.isdir(os.path.join(ImgPath, filename))):
                continue

            img = img_as_float(io.imread(os.path.join(ImgPath, filename)))
            img = transform.resize(img, (1024,1024))
            img = np.expand_dims(img, -1)
            Imgs.append(img)
            FileNames.append(filename)

            Imgs_mean += img.mean()
            Imgs_meansq += (img**2).mean()

        Imgs = np.array(Imgs)
        FileNames = np.array(FileNames)

        Imgs_mean /= len(FileNames)
        Imgs_meansq /= len(FileNames)
        Imgs_std  = np.sqrt(Imgs_meansq - Imgs_mean**2)

        Imgs -= Imgs_mean
        Imgs /= Imgs_std

        print("Preprocessing done on {} files.....".format(len(FileNames)))
        return Imgs, FileNames
    

    
    def predict(self, ImgPath):
        """
        Lung segmentation
        
        Args:
            (string) path : file path of image
            
        Return:
            Lung mask saved in Mask_DIR folder 
        """
        Mask_DIR = ImgPath + "/LungMask"
        
        testData, filenames = self._preprocessing(ImgPath)
        testData = np.asarray(testData)
                
        if (len(testData.shape) == 3):
            testData = np.expand_dims(testData, -1)
        if (not os.path.isdir(Mask_DIR)):
            os.makedirs(Mask_DIR)
            
        preds = self.model.predict(testData, batch_size=1)
        preds = preds > 0.5
        processed = _postprocessing(preds)

        for i in range(len(processed)):
            mask = processed[i, :, :, 0]
            mask = mask * 255
            cv2.imwrite(Mask_DIR + "/" + filenames[i], mask)

        del self.model
        gc.collect()
        K.clear_session()
        print(" Test Done ! ")
        
        return None