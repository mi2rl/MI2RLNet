import os
import sys
import cv2
import argparse
import numpy as np
import nibabel as nib

sys.path.append('../')

from base import BaseModule
from utils import Checker
from Kidney import KidneyTumorSegmentation

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',       type=str, default=None)
    parser.add_argument('--img',        type=str, default=None)
    parser.add_argument('--weights',    type=str, default=None)
    parser.add_argument('--save_path',  type=str, default=None)
    parser.add_argument('--gpus',       type=str, default='-1')
    return parser.parse_args()


def main(args):
    Checker.set_gpu(args.gpus, 'tf2')
    img_path = os.path.abspath(args.img)
    tumor_segmentation = KidneyTumorSegmentation()
    tumor_segmentation.init(args.mode)
    result = liver_segmentation.predict(img_path)
    
    if args.save_path is not None:
        
 
        
        Checker.set_gpu(args.gpus, 'tf2')
        tumor_segmentation = KidneyTumorSegmentation()
        tumor_segmentation.init(args.mode)
        
        result = liver_segmentation.predict(mode,img_path)
        
        if args.mode == '1':
            x_nib = nib.load(img_path)
            p_nib = nib.Nifti1Image(result[-1::-1], x_nib.affine)
            nib.save(p_nib, os.path.join('./result', args.mode, 'prediction_'+img_path.split('_')[1][:5]+'.nii'))
        
        else:
            if args.mode == '2_1':
                x_nib = nib.load(img_path)
                p_nib = nib.Nifti1Image(result[-1::-1], x_nib.affine)
                nib.save(p_nib, os.path.join('./result', args.mode, 'prediction_'+img_path.split('_')[1][:5]+'.nii'))
            else: 
                img_pair = nib.Nifti1Pair(result, np.diag([-spacing[0], spacing[1], spacing[2], 1]))
                nib.save(img_pair, os.path.join('./result', args.mode, 'prediction_'+img_path.split('_')[1][:5]+'.nii'))
                
        print(result.shape, type(result))
        
  


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)