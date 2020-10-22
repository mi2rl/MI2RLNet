"""
Brain Moduel Test Code
- blackblood segmentation
- [mra] brain extraction
"""


import numpy as np
import cv2
import argparse
import os
import sys

import SimpleITK as sitk

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--img', type=str, default=None)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    return parser.parse_args()


def main(args):
    ### For Preprocessing
    dcm_path = os.path.abspath(args.img)

    ### MRA_BET Example 
    if args.mode == 'mra_bet':
        from medimodule.Brain import MRA_BET
        mra_bet = MRA_BET()
        mra_bet.init(args.weights, gpu_num=3)
        out = mra_bet.predict(dcm_path)
        print(out)

    elif args.mode == 'blackblood_segmentation':
        from medibodule.Brain import BlackbloodSegmentation
        #TODO : 아래 테스트 코드를 작성해주세요.


if __name__ == '__main__':
   argv = parse_arguments(sys.argv[1:])
   main(argv)
    
