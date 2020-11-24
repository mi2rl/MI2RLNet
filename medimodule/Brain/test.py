"""
Brain Moduel Test Code
- blackblood segmentation
- [mra] brain extraction
"""

import argparse
import os
import sys
sys.path.append("../../")

import numpy as np
import cv2
import SimpleITK as sitk
from medimodule.utils import Checker
import warnings
import nibabel as nib

warnings.filterwarnings('ignore', '.*output shape of zoom.*')
warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--img', type=str, default=None)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--gpu', type=str, default=None)
    return parser.parse_args()


def main(args):
    ### For Preprocessing
    dcm_path = os.path.abspath(args.img)
    check = Checker()

    ### MRA_BET Example
    if args.mode == 'mra_bet':
        from medimodule.Brain.module import MRA_BET
        check.check_input_type(args.img, 'nii')
        check.set_gpu(gpu_idx=args.gpu, framework='pytorch')
        mra_bet = MRA_BET()
        mra_bet.init(args.weights)
        out = mra_bet.predict(dcm_path, save_path=args.save_path)
        print(out)

    ### Blackblood segmentation Example
    elif args.mode == 'blackblood_segmentation':
        from medimodule.Brain.module import BlackbloodSegmentation
        check.set_gpu(gpu_idx=args.gpu, framework='tf2')
        blackblood_segmentation = BlackbloodSegmentation()
        blackblood_segmentation.init(args.weights)
        out = blackblood_segmentation.predict(dcm_path)
        if args.save_path is not None:
            temp2 = np.swapaxes(out, 1, 2)
            temp2 = np.swapaxes(temp2, 0, 1)
            temp2 = np.swapaxes(temp2, 1, 2)
            mask_pair = nib.Nifti1Pair(temp2, np.diag([-blackblood_segmentation.space[0],
                                                      blackblood_segmentation.space[1],
                                                      blackblood_segmentation.space[2], 1]))
            nib.save(mask_pair, args.save_path)
        print(out.shape, out.dtype)

if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    main(argv)


