import os
import sys
import cv2
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',       type=str, default=None)
    parser.add_argument('--img',        type=str, default=None)
    parser.add_argument('--weights',    type=str, default=None)
    parser.add_argument('--save_path',  type=str, default=None)
    parser.add_argument('--gpus',       type=str, default='-1')
    return parser.parse_args()


def main(args):
    from medimodule.utils import Checker
    img_path = os.path.abspath(args.img)
    
    if args.mode == 'liver_segmentation':
        ### Liver Segmentation
        from medimodule.Liver import LiverSegmentation
        Checker.set_gpu(args.gpus, 'tf2')
        liver_segmentation = LiverSegmentation()
        liver_segmentation.init(os.path.abspath(args.weights))
        result = liver_segmentation.predict(img_path)
        if args.save_path is not None:
            import numpy as np
            import nibabel as nib
            temp2 = np.swapaxes(result, 1, 2)
            temp2 = np.swapaxes(temp2, 0, 1)
            temp2 = np.swapaxes(temp2, 1, 2)
            mask_pair = nib.Nifti1Pair(
                temp2, np.diag([-liver_segmentation.space[0], 
                                -liver_segmentation.space[1], 
                                5., 1]))
            nib.save(mask_pair, args.save_path)
        print(result.shape, type(result))


if __name__ == '__main__':
    # If running test.py in the module, add the root path.
    sys.path.append('../../')
    args = parse_arguments(sys.argv[1:])
    main(args)
    
