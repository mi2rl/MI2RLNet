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

    
    ### Age Regressor Example
    if args.mode == 'age_regression':
        from medimodule.Chest import AgeRegressor
        age_regressor = AgeRegressor()
        age_regressor.init(args.weights)
        out = age_regressor.predict(dcm_path)
        print(out)


    ### Viewpoint Classifier Example (PA / Lateral / Others)
    elif args.mode == 'viewpoint_classification':
        from medimodule.Chest import ViewpointClassifier
        view_classifier = ViewpointClassifier()
        view_classifier.init(args.weights)
        out = view_classifier.predict(dcm_path)
        print(out)

    ### Enhance Classifier Example (Non-Enhanced / Enhanced)
    elif args.mode == 'enhance_classification':
        from medimodule.Chest import EnhanceCTClassifier
        enhanceCT_classifier = EnhanceCTClassifier()
        enhanceCT_classifier.init(args.weights)
        out = enhanceCT_classifier.predict(dcm_path)
        print(out)
    elif args.mode == 'polyp_segmentation':
        from medimodule.Polyp import PolypSegmentation
        polyp_seg = PolypSegmentation()
        polyp_seg.init(args.weights)
        out = polyp_seg.predict(dcm_path)
        print(np.unique(out))
    ### Example LR Detection (L / R)
    elif args.mode == 'lr_detection':
        from medimodule.Chest import ChestLRDetection
        detection = ChestLRDetection()
        detection.init(args.weights)
        predict = detection.predict(args.img)
        cv2.imwrite(args.save_path + 'output.png' ,predict)
        print("Predict Done")
        print("="*30)

if __name__ == '__main__':
   argv = parse_arguments(sys.argv[1:])
   main(argv)
    