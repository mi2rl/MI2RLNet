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

    if args.mode == 'polyp_segmentation':
        from medimodule.Polyp import PolypSegmentation
        polyp_seg = PolypSegmentation()
        polyp_seg.init(args.weights)
        out = polyp_seg.predict(dcm_path)
        print(np.unique(out))

if __name__ == '__main__':
   argv = parse_arguments(sys.argv[1:])
   main(argv)
    
