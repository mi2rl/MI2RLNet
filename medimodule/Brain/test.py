"""
Brain Moduel Test Code
- blackblood segmentation
- [mra] brain extraction
"""

import argparse
import os
import sys
sys.path.append("../")
import numpy as np
import cv2
import SimpleITK as sitk
from utils import Checker
import warnings

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
        from Brain.module import MRA_BET
        check.check_input_type(args.img, 'nii')
        check.set_gpu(gpu_idx=args.gpu, framework='pytorch')
        mra_bet = MRA_BET()
        mra_bet.init(args.weights)
        out = mra_bet.predict(dcm_path, save_path=args.save_path)
        print(out)

    ### Blackblood segmentation Example
    elif args.mode == 'blackblood_segmentation':
        from Brain.module import BlackbloodSegmentation

        check.check_input_type(args.img, 'nii')
        check.set_gpu(gpu_idx=args.gpu, framework='tf2')

        blackblood_segmentation = BlackbloodSegmentation()
        blackblood_segmentation.init(args.weights)
        out = blackblood_segmentation.predict(dcm_path)
        print(out.shape, type(out), out)

if __name__ == '__main__':
   argv = parse_arguments(sys.argv[1:])
   main(argv)