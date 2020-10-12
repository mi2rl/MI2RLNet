import numpy as np
import cv2
import argparse
import os
import sys

import SimpleITK as sitk

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default=None)

    return parser.parse_args()


def main(args):
    ### For Preprocessing
    dcm_path = 'test.dcm'
    # load and convert
    img = sitk.GetArrayFromImage(sitk.ReadImage(dcm_path))
    img = img[0]

    img = cv2.resize(img, (512,512), interpolation=cv2.INTER_LINEAR)

    np_img = img.astype(np.float32)
    np_img -= np.min(np_img)
    np_img /= np.percentile(np_img, 99)

    np_img[np_img>1] = 1
    np_img *= (2**8-1)
    np_img = np_img.astype(np.uint8)

    narr = np.expand_dims(np.expand_dims(np_img, 0), -1)

    ### Age Regressor Example
    if args.mode == 'age_regression':
        from medimodule.Chest import AgeRegressor
        age_regressor = AgeRegressor()
        age_regressor.init('weights/age_regression.h5')
        out = age_regressor.predict(dcm_path)
        print(out)


    ### Viewpoint Classifier Example (PA / Lateral / Others)
    elif args.mode == 'viewpoint_classification':
        from medimodule.Chest import ViewpointClassifier
        view_classifier = ViewClassifier()
        view_classifier.init('weights/viewpoint_classification.h5')
        out = view_classifier.predict(dcm_path)
        print(out)

    ### Enhance Classifier Example (Non-Enhanced / Enhanced)
    elif args.mode == 'classification_enhance':
        from medimodule.Chest import EnhanceCTClassifier
        enhanceCT_classifier = EnhanceCTClassifier()
        enhanceCT_classifier.init('weights/enhance_classification.h5')
        out = enhaceCT_classifier.predict(dcm_path)
        print(out)


if __name__ == '__main__':
   argv = parse_arguments(sys.argv[1:])
   main(argv)
