import os
import sys
import cv2
import warnings
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from typing import Optional, Tuple

from medimodule.utils import Checker

warnings.filterwarnings('ignore', '.*output shape of zoom.*')
warnings.filterwarnings("ignore", category=DeprecationWarning)


def BrainBET(
    task: str,
    weight: str,
    image_path: str,
    save_path: Optional[str] = None,
    gpus: str = "-1"
) -> Tuple[np.array, np.array]:
    """
    """

    from medimodule.Brain import MRI_BET
    Checker.check_input_type(image_path, "nii")
    Checker.set_gpu(gpu_idx=gpus, framework="pytorch")

    model = MRI_BET(weight)
    image, mask = model.predict(
        os.path.abspath(image_path),
        img_type="T1" if "mri" in task else "MRA",
        save_path=save_path)

    return image, mask


def BrainBlackBloodSegmentation(
    task: str,
    weight: str,
    image_path: str,
    save_path: Optional[str] = None,
    gpus: str = "-1"
) -> Tuple[np.array, np.array]:
    """
    """

    from medimodule.Brain import BlackbloodSegmentation
    Checker.check_input_type(image_path, "nii")
    Checker.set_gpu(gpu_idx=gpus, framework="tf")

    model = BlackbloodSegmentation(weight)
    image, mask = model.predict(
        os.path.abspath(image_path),
        save_path=save_path)

    return image, mask


def ChestLRmarkDetection(
    task: str,
    weight: str,
    image_path: str,
    save_path: Optional[str] = None,
    gpus: str = "-1"
) -> Tuple[np.array, dict]:
    """
    """

    from medimodule.Chest import LRmarkDetection
    Checker.check_input_type(image_path, ["png", "jpg", "bmp"])
    Checker.set_gpu(gpu_idx=gpus, framework="tf")

    model = LRmarkDetection(weight)
    image, result = model.predict(
        os.path.abspath(image_path),
        save_path=save_path)

    return image, result


def ChestViewpointClassifier(
    task: str,
    weight: str,
    image_path: str,
    save_path: Optional[str] = None,
    gpus: str = "-1"
) -> Tuple[np.array, str]:
    """
    """

    from medimodule.Chest import ViewpointClassifier
    Checker.check_input_type(image_path, ["dcm"])
    Checker.set_gpu(gpu_idx=gpus, framework="tf")

    model = ViewpointClassifier(weight)
    image, result = model.predict(
        os.path.abspath(image_path),
        save_path=save_path)

    return image, result


def ChestEnhanceCTClassification(
    task: str,
    weight: str,
    image_path: str,
    save_path: Optional[str] = None,
    gpus: str = "-1"
) -> Tuple[np.array, str]:
    """
    """

    from medimodule.Chest import EnhanceCTClassifier
    Checker.check_input_type(image_path, ["dcm"])
    Checker.set_gpu(gpu_idx=gpus, framework="tf")

    model = EnhanceCTClassifier(weight)
    image, result = model.predict(
        os.path.abspath(image_path),
        save_path=save_path)

    return image, result


def ChestLungSegmentation(
    task: str,
    weight: str,
    image_path: str,
    save_path: Optional[str] = None,
    gpus: str = "-1"
) -> Tuple[np.array, np.array]:
    """
    """

    from medimodule.Chest import LungSegmentation
    Checker.check_input_type(image_path, ["png", "jpg"])
    Checker.set_gpu(gpu_idx=gpus, framework="tf")

    model = LungSegmentation(weight)
    image, mask = model.predict(
        os.path.abspath(image_path),
        save_path=save_path)

    return image, mask


def AbdomenLiverSegmentation(
    task: str,
    weight: str,
    image_path: str,
    save_path: Optional[str] = None,
    gpus: str = "-1"
) -> Tuple[np.array, np.array]:
    """
    """

    from medimodule.Liver import LiverSegmentation
    Checker.check_input_type(image_path, ["hdr", "img", "nii"])
    Checker.set_gpu(gpu_idx=gpus, framework="tf")

    model = LiverSegmentation(weight)
    image, mask = model.predict(
        os.path.abspath(image_path),
        save_path=save_path)

    return image, mask


def main(args):
    Checker.check_args(args.part, args.task)
    input_kwargs = dict(
        task=args.task,
        weight=args.weight,
        image_path=args.image_path,
        save_path=args.save_path,
        gpus=args.gpus)
    
    if args.part == "Brain":
        if args.task in ["mribet", "mrabet"]:
            BrainBET(**input_kwargs)
        elif args.task == "blackblood_segmentation":
            BrainBlackBloodSegmentation(**input_kwargs)

    elif args.part == "Chest":
        if args.task == "lung_segmentation":
            ChestLungSegmentation(**input_kwargs)
        elif args.task == "lrmark_detection":
            ChestLRmarkDetection(**input_kwargs)
        elif args.task == "viewpoint_classification":
            ChestViewpointClassifier(**input_kwargs)
        elif args.task == "enhance_classification":
            ChestEnhanceCTClassification(**input_kwargs)

    elif args.part == "Abdomen":
        if args.task == "liver_segmentation":
            AbdomenLiverSegmentation(**input_kwargs)

        elif args.task == "kidney_tumor_segmentation":
            pass
            # AbdomenKidneyTumorSegmentation(**input_kwargs)

    elif args.part == "Colon":
        pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part",       type=str,   default=None, required=True,
                        help="[Brain / Chest / Abdomen / Colon]")
    parser.add_argument("--task",       type=str,   default=None, required=True,
                        help="Choose a specific part to be executed.")
    parser.add_argument("--weight",     type=str,   default=None)
    parser.add_argument("--image_path", type=str,   default=None, required=True)
    parser.add_argument("--save_path",  type=str,   default=None)
    parser.add_argument("--gpus",       type=str,   default="-1")

    args = parser.parse_args()

    main(args)
