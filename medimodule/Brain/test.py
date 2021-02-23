"""
Brain Moduel Test Code
- blackblood segmentation
- [mri] brain extraction
"""

import argparse
import os
import sys

import numpy as np
import cv2
import SimpleITK as sitk
import warnings
import nibabel as nib


sys.path.append(os.getcwd())
from utils import Checker

warnings.filterwarnings('ignore', '.*output shape of zoom.*')
warnings.filterwarnings("ignore", category=DeprecationWarning)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--img', type=str, default=None)
    parser.add_argument('--img_type', type=str, default='T1', help='T1/MRA')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--save_mask', type=bool, default=False)
    parser.add_argument('--save_stripping', type=bool, default=False)
    parser.add_argument('--save_path',  type=str, default=None)

    return parser.parse_args()


def main(args):
    ### For Preprocessing
    nii_path = os.path.abspath(args.img)
    check = Checker()

    ### MRI_BET Example
    if args.mode == 'mri_bet':
        from Brain.module import MRI_BET
        check.check_input_type(args.img, 'nii')
        check.set_gpu(gpu_idx=args.gpu, framework='pytorch')
        mri_bet = MRI_BET()
        mri_bet.init(args.weights)
        out = mri_bet.predict(nii_path, 
                            img_type=args.img_type,
                            save_mask=args.save_mask, 
                            save_stripping=args.save_stripping)
        print(out)

    ### Blackblood segmentation Example
    elif args.mode == 'blackblood_segmentation':
        from Brain.module import BlackbloodSegmentation

        check.check_input_type(args.img, 'nii')
        check.set_gpu(gpu_idx=args.gpu, framework='tf2')

        blackblood_segmentation = BlackbloodSegmentation()
        blackblood_segmentation.init(args.weights)
        out = blackblood_segmentation.predict(nii_path)
        if args.save_path is not None:
            temp2 = np.swapaxes(out, 1, 2)
            temp2 = np.swapaxes(temp2, 0, 1)
            temp2 = np.swapaxes(temp2, 1, 2)
            mask_pair = nib.Nifti1Pair(temp2, np.diag([-blackblood_segmentation.space[0],
                                                      blackblood_segmentation.space[1],
                                                      blackblood_segmentation.space[2], 1]))
            nib.save(mask_pair, args.save_path)
        print(out.shape, type(out))


if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    main(argv)


