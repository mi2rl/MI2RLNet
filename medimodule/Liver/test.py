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

    
    ### Liver Segmentation
    # TODO : edit below codes according to your codes
    if args.mode == 'age_regression':
        from medimodule.Chest import AgeRegressor
        age_regressor = AgeRegressor()
        age_regressor.init(args.weights)
        out = age_regressor.predict(dcm_path)
        print(out)



if __name__ == '__main__':
   argv = parse_arguments(sys.argv[1:])
   main(argv)
    
