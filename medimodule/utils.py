import os
import sys


class Checker:

    def check_input_type(self, _input, _t):
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
        assert input_ext == _t, 'expected {} file extension, but got {} file extension'.format(_t, input_ext)


    def set_gpu(self, gpu_idx, framework):
        """
        Set a gpu 

        Args:
            (string) gpu_idx : gpu index for specific gpu allocation
            (string) framework : tensorflow / pytorch / ...

        Return:
            If gpu is available, set the gpu
        """
        
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx

            

