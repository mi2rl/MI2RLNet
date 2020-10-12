import numpy as np
import cv2
import argparse
import os
import sys

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default=None)

    return parser.parse_args()


def main(args):
    ### For Preprocessing
    dcm_path = 'test.dcm'

    from medimodule.preprocessing.dicom_handler import XrayHandler
    xray_handler = XrayHandler()
    narr = xray_handler.dicom_to_numpy(dcm_path, out_size=(512,512), out_channels=1)
    # for keras shape
    narr = np.expand_dims(np.expand_dims(narr, 0), -1)

    ### Age Regressor Example
    if args.mode == 'age_regression':
        from medimodule.chest import AgeRegressor
        age_regressor = AgeRegressor()
        age_regressor.init('weights/age_regression.h5')
        out = age_regressor.predict(dcm_path)
        print(out)


    ### Viewpoint Classifier Example (PA / Lateral / Others)
    elif args.mode == 'viewpoint_classification':
        from medimodule.chest import ViewpointClassifier
        view_classifier = ViewClassifier()
        view_classifier.init('weights/viewpoint_classification.h5')
        out = view_classifier.predict(dcm_path)
        print(out)

    ### Enhance Classifier Example (Non-Enhanced / Enhanced)
    elif args.mode == 'classification_enhance':
        from medimodule.chest import EnhanceCTClassifier
        enhanceCT_classifier = EnhanceCTClassifier()
        enhanceCT_classifier.init('weights/enhance_classification.h5')
        out = enhaceCT_classifier.predict(dcm_path)
        print(out)


if __name__ == '__main__':
   argv = parse_arguments(sys.argv[1:])
   main(argv)
