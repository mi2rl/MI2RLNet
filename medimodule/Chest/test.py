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

    ### Viewpoint Classifier Example (PA / Lateral / Others)
    elif args.mode == 'viewpoint_classification':
        from medimodule.Chest.module import ViewpointClassifier
        view_classifier = ViewpointClassifier()
        view_classifier.init(args.weights)
        out = view_classifier.predict(dcm_path)
        print(out)

    ### Enhance Classifier Example (Non-Enhanced / Enhanced)
    elif args.mode == 'enhance_classification':
        from medimodule.Chest.module import EnhanceCTClassifier
        enhanceCT_classifier = EnhanceCTClassifier()
        enhanceCT_classifier.init(args.weights)
        out = enhanceCT_classifier.predict(dcm_path)
        print(out)

    ### Example LR Detection (L / R)
    elif args.mode == 'lr_detection':
        from medimodule.Chest.module import ChestLRDetection
        detection = ChestLRDetection()
        detection.init(args.weights)
        predict = detection.predict(args.img)


if __name__ == '__main__':
   argv = parse_arguments(sys.argv[1:])
   main(argv)




