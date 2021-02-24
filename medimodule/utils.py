import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import logging
import warnings
from typing import List

modules = {
    'Chest' : [
        'lrmark_detection',
        'viewpoint_classification',
        'enhance_classification'
    ],
    'Brain' : [
        'mribet',
        'mrabet',
        'blackblood_segmentation'
    ],
    'Abdomen': [
        'liver_segmentation'
    ]
}

class Checker:

    @staticmethod
    def check_input_type(_input: str, _t: List[str]):
        """
        Check a input image type.

        Args: 
            (string) _input : input file path or name
            (string) _t : image type (dcm, png, nii, hdr, ...) 
            !!! please remove point in _t (ex: .dcm --> dcm) !!!

        Return:
            If there is no problem, it doesn't show anything.
            However, if there are some problems, it will makes a assertion error.
        """
        input_ext = _input.split('.')[-1]
        assert input_ext in _t, 'expected {} file extension, but got {} file extension'.format(_t, input_ext)

    @staticmethod
    def check_input_type_bool(_input: str, _t: List[str]) -> bool:
        """
        Check a input image type and return bool.

        Args:
            (string) _input : input file path or name
            (string) _t : image type (dcm, png, nii, hdr, ...) 
            !!! please remove point in _t (ex: .dcm --> dcm) !!!

        Return:
            (bool) Whether the correct format or not
        """
        input_ext = _input.split('.')[-1]
        if input_ext in _t:
            return True
        else:
            return False

    @staticmethod
    def set_gpu(gpu_idx=None, framework=None):
        """
        Set specific device(s) and allow growth.

        Args:
            (string) gpu_idx : gpu index for specific gpu allocation
            (string) framework : tf1 / keras_tf1 / tf2 / pytorch

        Return:
            If gpu is available, set the gpu
        """

        assert gpu_idx is not None, 'gpu_idx must be selected.'
        assert isinstance(gpu_idx, str)
        assert framework is not None, 'framework must be selected.'
        assert framework in ['tf', 'pytorch'], \
            'This module supports these frameworks, tf and pytorch.'
        
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
        if gpu_idx == '-1':
            warnings.warn('Running time may be long in cpu mode.')
        else:
            if 'tf' == framework:
                # only tensorflow 2.x
                import tensorflow as tf
                assert int(tf.__version__.split('.')[0]) >= 2.0
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    try:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        print(f"Complete to allocate gpu(s) at {gpu_idx} and set memory growth in TF2.")
                    except RuntimeError as e:
                        # Memory growth must be set before GPUs have been initialized
                        print(e)
            else:
                # pytorch
                print(f"Complete to allocate gpu(s) at {gpu_idx}.")
                            
    @staticmethod
    def check_args(part, task):
        if part not in modules.keys():
            msg = 'part: {}, task:{} is not implemented, implemented modules (key: part, value: tasks) as follows :\n {}\n'.format(
                part, task, modules)
            raise NotImplementedError(msg)
