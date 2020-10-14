import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
import cv2
from medimodule.base import BaseModule
from medimodule.Chest.LR_detection.lr_detection.load_model import build_lr_detection
from medimodule.Chest.LR_detection.lr_detection.utils.anchors import anchors_for_shape
from medimodule.Chest.LR_detection.lr_detection.utils.draw_boxes import draw_boxes
from medimodule.Chest.LR_detection.lr_detection.utils.post_process_boxes import post_process_boxes

class ChestLRDetection(BaseModule):
    def init(self, weight_path,gpu_num , score_threshold):
        """
        Initialize the model with its weight.
        
        Args:
            (string) weight_path : model's weight path
            (string) gpu_num : select GPU number
        """

        self.model = build_lr_detection(weight_path , gpu_num,score_threshold)
        
    def _preprocessing(self, path):
        """
        Preprocess the image from the path

        Args:
            (string) path : absolute path of image
        Return:
            (numpy ndarray) new_image : scaled image
            (numpy ndarray) src_image : origin_image
            (float) scale, offset_h, offset_w , image_size , h,w : scaled image informations
        """
        
        '''
        TODO : check image format
            - hdr : pass
            - nii : ?
            - dcm : ?
            - png : image preprocessing
        '''

        if '.nii' in path:
            raise NotImplementedError('TODO : .nii is not clear about horizontal or vertical shape.')

        elif '.dcm' in path:
            raise ValueError('.dcm is not supported. Please convert dcm dummies to analyze format.')

        mean_std = [29.311405133024834, 43.38181786843102]
        image = sitk.ReadImage(path)
        space = image.GetSpacing()
        image = np.squeeze(sitk.GetArrayFromImage(image).astype('float32')) # (d, w, h)
        d, w, h = image.shape
        image = zoom(image, [space[-1]/5., 256./float(w), 256./float(h)], order=1, mode='constant')
        image = np.clip(image, 10, 190)
        image = (image - mean_std[0]) / mean_std[1]
        image = image[np.newaxis,...,np.newaxis] # (1, d, w, h, 1)
        
        if '.png' in path or '.jpg' in path or '.bmp' in path:
            
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            src_image = image.copy()
            image = image[:, :, ::-1]
            h, w = image.shape[:2]
            
            image_height, image_width = image.shape[:2]
            image_size = 512 # b0 기준 
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
        
        return new_image, scale, offset_h, offset_w , image_size , src_image, h,w

    def predict(self, path,score_threshold):
        """
        L,R Detection 

        Args:
            (string) path : 8-bit png path
            
        Return:
            (numpy ndarray) L,R detection image (W,H,C)
        """
        classes = ['left','right']
        num_classes = len(classes)
        colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
        
        inner_threshold = score_threshold
        image, scale, offset_h, offset_w ,image_size ,src_image,h,w= self._preprocessing(path)
        
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
    
        indices = np.where(scores[:] > inner_threshold)[0]
        boxes = boxes[indices]
        labels = labels[indices]

        draw_boxes(src_image, boxes, scores, labels, colors, classes)

        return (src_image)