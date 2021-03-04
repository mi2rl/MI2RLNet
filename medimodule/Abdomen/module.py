import os
import cv2
import warnings
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import zoom
from typing import Tuple, Optional, List

import torch
import tensorflow as tf

from medimodule.utils import Checker
from medimodule.base import BaseModule
from medimodule.Abdomen.models import LiverSeg
from medimodule.Abdomen.models import PolypDet
from medimodule.Abdomen.models import KidneyTumorSeg
from medimodule.Abdomen.models import KidneyUtils


class LiverSegmentation(BaseModule):
    def __init__(self, weight_path: Optional[str] = None):
        """
        Initialize the model with its weight.
        
        Args:
            (string) weight_path : model's weight path
        """

        self.model = LiverSeg()
        if weight_path is not None:
            self.model.load_weights(weight_path)

    def _preprocessing(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the image from the path

        Args:
            (string) path : absolute path of image
        Return:
            (numpy ndarray) image
        """

        mean_std = [29.311405133024834, 43.38181786843102]
        if Checker.check_input_type_bool(path, 'nii'):
            image = sitk.ReadImage(path)
            self.space = image.GetSpacing()
            image = sitk.GetArrayFromImage(image).astype('float32')
            warnings.warn(
                '.nii is not recommended as an image format '
                'due to be not clear abour horizontal or vertical shape. '
                'Please check the sample in README.md.', UserWarning)

        elif Checker.check_input_type_bool(path, 'dcm'):
            raise ValueError(
                '.dcm is not supported. '
                'Please convert dcm dummies to analyze format.')

        elif Checker.check_input_type_bool(path, 'img') or \
            Checker.check_input_type_bool(path, 'hdr'):
            image = sitk.ReadImage(path)
            self.space = image.GetSpacing()
            image = np.squeeze(sitk.GetArrayFromImage(image).astype('float32')) # (d, w, h)

        elif Checker.check_input_type_bool(path, 'npy'):
            image = np.load(path)
            self.space = [1., 1., 1.]
            warnings.warn(
                '.npy is not recommended as an image format.'
                'Since spacing cannot be identified from .npy, '
                'spacing is set as [1., 1., 1.].', 
                UserWarning)

        else:
            input_ext = path.split('.')[-1]
            raise ValueError(
                f'.{input_ext} format is not supported.')

        self.img_shape = image.shape
        _, h, w = self.img_shape

        imageo = image.copy()
        image = zoom(
            image, [self.space[-1]/5., 256./float(w), 256./float(h)], 
            order=1, mode='constant')
        image = np.clip(image, 10, 190)
        image = (image - mean_std[0]) / mean_std[1]
        image = image[np.newaxis,...,np.newaxis] # (1, d, w, h, 1)
        return imageo, image

    def predict(
        self, 
        path: str, 
        save_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Liver segmentation
        Args:
            (string) path : image path (hdr/img, nii, npy)
            (string) save_path : 
            (bool) istogether: with image which was used or not

        Return:
            (numpy ndarray) liver mask with shape (depth, width, height)
        """

        path = os.path.abspath(path)
        imgo, img = self._preprocessing(path)
        mask = np.squeeze(self.model(img).numpy().argmax(axis=-1))
        mask_shape = mask.shape
        mask = zoom(mask, [self.img_shape[0]/mask_shape[0], 
                           self.img_shape[1]/mask_shape[1], 
                           self.img_shape[2]/mask_shape[2]],
                    order=1, mode='constant').astype(np.uint8)

        if save_path:
            temp2 = np.swapaxes(mask, 1, 2)
            temp2 = np.swapaxes(temp2, 0, 1)
            temp2 = np.swapaxes(temp2, 1, 2)
            mask_pair = nib.Nifti1Pair(temp2, np.diag([-self.space[0], -self.space[1], 5., 1]))
            nib.save(mask_pair, save_path)

        return (np.squeeze(imgo), mask)


class PolypDetection(BaseModule):
    def __init__(self, weight_path: Optional[str] = None):
        """
        Initialize the model with its weight.
        
        Args:
            (string) weight_path : model's weight path
        """

        self.model = PolypDet()
        if weight_path:
            weight = torch.load(weight_path, map_location="cpu")
            self.model.load_state_dict(weight["net"])

    def _preprocessing(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the image from the path
        Args:
            (string) path : absolute path of image
        Return:
            (numpy ndarray) image
        """
        
        imageo = cv2.imread(path)
        image = cv2.resize(imageo, (512, 512)).astype(np.float32)
        image /= 255.
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        return imageo, image

    def predict(
        self, 
        path: str, 
        save_path: Optional[str] = None, 
        thresh: float = 0.7
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Liver segmentation
        Args:
            (string) path : image path
            (string) save_path : 
            (bool) thresh : the value of pixel which you will start to use as polyp Lesions(0 ~ 1). 
                            if it is 0.7, the pixel which has value under 0.7 won't be used as Lesion pixel.
        Return:
            (numpy ndarray) polyp mask with shape (width, height)
        """

        fn_thresh = lambda x, thresh :  1.0 * (x > thresh)

        imgo, img = self._preprocessing(path)
        mask = self.model(img) 
        mask = fn_thresh(mask, thresh)
        mask = np.squeeze(mask.numpy())
        mask = cv2.resize(mask, imgo.shape[:2])
        mask = mask.astype(np.uint8)
        
        if save_path:
            cv2.imwrite(save_path, mask * 255)

        return imgo, mask


class KidneyTumorSegmentation(BaseModule):
    """Kidney and Tumor Segmentation model with KiTS19 Challenge in MICCAI 2019.

    This model is implemented as cascaded segmentation of mode 1 and mode 2_1~5.
    mdoe 1 and mode 2_1 was implemented by coreline soft.
    mode 2_2~5 was implemented by MI2RL in Asan Medical Center.
    mode 1 : detecting kidney's location in whole abdomen CT using semantic segmentation
    mode 2 : segmenting kidney and tumor in detected the location
    """
    def __init__(self, weight_path: List[str] = None):
        """
        Initialize the model with its weight.
        
        Args:
            (string) weight_path : model's weight path
        """

        self.config = {
            "1": {
                'depth': 3,
                'wlower': -300,
                'wupper': 600,
                'input_dim': (200, 200, 200),
                'num_labels_1ststg': 1}, 
            "2_1": {
                'depth': 3,
                'wlower': -300,
                'wupper': 600,
                'input_dim': (200, 200, 200)},
            "2_2": {
                'lossfn': 'dice',
                'depth': 4,
                'standard': 'normal',
                'task': 'tumor',
                'wlower': -100,
                'wupper': 300},
            "2_3": {
                'lossfn': 'dice',
                'depth': 3,
                'standard': 'minmax',
                'task': 'tumor1',
                'wlower': -100,
                'wupper': 300},
            "2_4": {
                'lossfn': 'focaldice',
                'depth': 3,
                'standard': 'minmax',
                'task': 'tumor1',
                'wlower': -100,
                'wupper': 300},
            "2_5": {
                'lossfn': 'dice',
                'depth': 3,
                'standard': 'normal',
                'task': 'tumor1',
                'wlower': -100,
                'wupper': 300}}

        self._load_model()
        if weight_path:
            for wp, m in zip(weight_path, ["1", "2_1", "2_2", "2_3", "2_4", "2_5"]):
                try:
                    self.model[m].load_weights(wp)
                    print(f"Load weights in mode {m} model!")
                except:
                    pass
            

    def _load_model(self):
        self.model = {
            "1": KidneyTumorSeg(
                input_shape=(None, None, None, 1),
                num_labels=1,
                base_filter=32,
                depth=self.config["1"]["depth"],
                se_res_block=True,
                se_ratio=16,
                last_relu=True),
            "2_1": KidneyTumorSeg(
                input_shape=(None, None, None, 1),
                num_labels=3,
                base_filter=32,
                depth=self.config["2_1"]["depth"],
                se_res_block=True,
                se_ratio=16,
                last_relu=False),
        }

    def _preprocessing(
        self, 
        mode: str,
        path: Optional[str] = None,
        imageo: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ):

        """
        Preprocess the image from the path.
        Args:
            (string) path : absolute path of image
        Return:
            (numpy ndarray) image
        """

        if path is None:
            assert imageo is not None and mask is not None
        else:
            assert imageo is None and mask is None

        if mode == "1":
            # coreline soft
            img_ct_sag = sitk.ReadImage(path)
            img_ct_axial = KidneyUtils.transaxis(img_ct_sag, dtype=np.int16)
            raw_ct = sitk.GetArrayFromImage(img_ct_axial)

            raw_ct_shape = np.shape(raw_ct)
            is_large_z = True if raw_ct_shape[0] > 200 else False

            if is_large_z:
                z_list = list(np.arange(0, raw_ct_shape[0] - 200, 100)) + [raw_ct_shape[0] - 200]
                x_start_src = 0
                x_end_src = int(raw_ct_shape[2] * 3 / 5)
                whole_imgs = []
                for i in range(2):
                    imgs = []
                    for z_start in z_list:
                        raw_ct_frame_shape = (200, raw_ct_shape[1], int(raw_ct_shape[2] * 3 / 5))
                        raw_ct_frame = np.ones(raw_ct_frame_shape, dtype=np.float32) * -1024
                        if i == 0:
                            # right
                            raw_ct_frame = raw_ct[z_start:z_start+200,:,x_start_src:x_end_src]
                        else:
                            # left
                            raw_ct_frame = raw_ct[z_start:z_start+200,:,-raw_ct_frame_shape[2]:]
                            raw_ct_frame = raw_ct_frame[:,:,-1::-1]

                        img_ct = sitk.GetImageFromArray(raw_ct_frame)
                        img_ct_rs = KidneyUtils.resample_img_asdim(img_ct, self.config["1"]["input_dim"], c_val=-1024)
                        raw_ct_rs = sitk.GetArrayFromImage(img_ct_rs)
                        raw_ct_rs_normed = KidneyUtils.normalize_vol(
                            raw_ct_rs, norm_wind_lower=self.config["1"]["wlower"], norm_wind_upper=self.config["1"]["wupper"])
                        raw_ct_rs_normed = raw_ct_rs_normed[None,...,None]
                        imgs.append(raw_ct_rs_normed)
                    imgs = np.concatenate(imgs, axis=0)
                    whole_imgs.append(imgs)

                return raw_ct, whole_imgs[0], whole_imgs[1], None, None, True # Tuple[np.ndarray, np.ndarray, np.ndarray, bool]
                
            else:
                z_start_dst = int((200 - raw_ct_shape[0]) / 2)
                z_end_dst = z_start_dst + raw_ct_shape[0]
                whole_imgs = []
                for i in range(2):
                    if i == 0:
                        # right
                        x_start_src = 0
                        x_end_src = int(raw_ct_shape[2] * 3 / 5)
                        raw_ct_frame_shape = (200, raw_ct_shape[1], int(raw_ct_shape[2] * 3 / 5))
                    else:
                        # left
                        x_start_src = int(raw_ct_shape[2] * 2 / 5)
                        x_end_src = raw_ct_shape[2]
                        raw_ct_frame_shape = (200, raw_ct_shape[1], x_end_src-x_start_src)

                    raw_ct_frame = np.ones(raw_ct_frame_shape, dtype=np.float32) * -1024
                    raw_ct_frame[z_start_dst:z_end_dst] = raw_ct[...,x_start_src:x_end_src]
                    if i == 1:
                        # left
                        raw_ct_frame = raw_ct_frame[...,-1::-1]

                    img_ct = sitk.GetImageFromArray(raw_ct_frame)
                    img_ct_rs = KidneyUtils.resample_img_asdim(img_ct, self.config["1"]["input_dim"], c_val=-1024)
                    raw_ct_rs = sitk.GetArrayFromImage(img_ct_rs)
                    raw_ct_rs_normed = KidneyUtils.normalize_vol(
                        raw_ct_rs, norm_wind_lower=self.config["1"]["wlower"], norm_wind_upper=self.config["1"]["wupper"])
                    raw_ct_rs_normed = raw_ct_rs_normed[None,...,None]
                    whole_imgs.append(raw_ct_rs_normed)

                return raw_ct, whole_imgs[0], whole_imgs[1], z_start_dst, z_end_dst, False # Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, bool]

        elif mode == "2_1":
            # coreline soft
            raise NotImplementedError()

        else:
            # MI2RL
            raise NotImplementedError()
            # config = self.config[mode]
            # if config["task"] == "tumor":
            #     cuts = [mask.copy() for _ in range(2)]
            #     cuts[0][int(cuts[0].shape[0]//2):] = 0
            #     cuts[1][:-int(cuts[1].shape[0]//2)] = 0

            #     if cuts[0].sum() == 0 or cuts[1].sum() == 0:
            #         return imageo, None
                
            #     coords = [np.where(c != 0) for c in cuts]
            #     bboxs = [[min(c[0]), max(c[0]),
            #               min(c[1]), max(c[1]),
            #               min(c[2]), max(c[2])] for c in coords]
                
            #     # crop
            #     image = [imageo[b[1]+1:b[0]:-1,b[2]:b[3]+1,b[4]:b[5]+1] 
            #              if config["task"] == "tumor" else 
            #              imageo[b[1]+1:b[0]:-1,b[2]:b[3]+1,b[4]:b[5]+1] 
            #              for b in bboxs]
            #     # windowing
            #     image = [np.clip(img, config["wlower"], config["wupper"]) for img in image]
            #     # standardization
            #     image = [(img-img.min())/(img.max()-img.min()) 
            #              if config["standard"] == "minmax" else
            #              (img-80.96544691699005)/58.84357050328374 
            #              for img in image]

        
        return imageo, image

    def predict(
        self, 
        path: str, 
        save_path: Optional[str] = None, 
        thresh: float = 0.7
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Liver segmentation
        Args:
            (string) path : image path
            (string) save_path : 
            (bool) thresh : the value of pixel which you will start to use as polyp Lesions(0 ~ 1). 
                            if it is 0.7, the pixel which has value under 0.7 won't be used as Lesion pixel.
        Return:
            (numpy ndarray) polyp mask with shape (width, height)
        """

        ##################################
        ############# mode 1 #############
        ##################################
        imgo, right, left, z_start, z_end, is_large_z = self._preprocessing(mode="1", path=path)

        raw_pred_whole = np.zeros_like(imgo)
        for i, img in enumerate([right, left]):
            # set shape
            if i == 0:
                # right
                raw_ct_shape = (imgo.shape[0], imgo.shape[1], int(imgo.shape[2] * 3 / 5))
            else:
                # left
                x_start_src = int(imgo.shape[2] * 2 / 5)
                x_end_src = imgo.shape[2]
                raw_ct_shape = (imgo.shape[0], imgo.shape[1], x_end_src-x_start_src)

            # predict masks
            if is_large_z:
                raw_pred_tmp = np.zeros(shape=(imgo.shape[0], 200, 200, self.config["1"]["num_labels_1ststg"]))
                raw_pred_tmp_cnt = raw_pred_tmp.copy()
                num_imgs = img.shape[0]
                for n in range(num_imgs):
                    raw_pred_tmp[z_start:z_start+200] += np.squeeze(self.model["1"].predict(np.expand_dims(img[n], axis=0)))
                    raw_pred_tmp_cnt[z_start:z_start+200] += 1

                raw_pred_tmp[np.where(raw_pred_tmp_cnt > 0)] /= raw_pred_tmp_cnt[np.where(raw_pred_tmp_cnt > 0)]
                raw_pred_tmp = np.squeeze(raw_pred_tmp)
                raw_pred_tmp[np.where(raw_pred_tmp > .5)] = 1
                raw_pred = sitk.GetArrayFromImage(
                    KidneyUtils.resample_img_asdim(sitk.GetImageFromArray(raw_pred_tmp), 
                    tuple(reversed(raw_ct_shape)), 
                    interp=sitk.sitkNearestNeighbor))
                raw_pred[np.where(raw_pred > 0.5)] = 1
                raw_pred = KidneyUtils.CCL_check_1ststg(raw_pred)

            else:
                result = self.model["1"](img)
                if result.shape[-1] == 1:
                    result = np.squeeze(result)
                else:
                    result = np.squeeze(np.argmax(result, axis=-1))

                result = result[z_start:z_end]
                raw_pred = sitk.GetArrayFromImage(
                    KidneyUtils.resample_img_asdim(sitk.GetImageFromArray(result),
                    tuple(reversed(raw_ct_shape)),
                    interp=sitk.sitkNearestNeighbor))
                raw_pred[np.where(raw_pred > .5)] = 1
                raw_pred = KidneyUtils.CCL_check_1ststg(raw_pred)

            # combine right and left
            if i == 0:
                # right
                raw_pred_whole[...,:raw_pred.shape[2]] = raw_pred
            else:
                # left
                raw_pred = raw_pred[...,-1::-1]
                raw_pred_whole_left_tmp = raw_pred_whole[...,-raw_pred.shape[2]:]
                raw_pred_whole_left_tmp[np.where(raw_pred > 0)] = raw_pred[np.where(raw_pred > 0)]
                raw_pred_whole[...,-raw_pred.shape[2]:] = raw_pred_whole_left_tmp

            mask_mode1 = KidneyUtils.CCL_check_1ststg(raw_pred_whole)
        
        if save_path:
            x_nib = nib.load(os.path.join(path))
            for suffix, mask in zip(["mask_mode1.nii"], [mask_mode1]):
                p_nib = nib.Nifti1Image(mask[-1::-1].astype(np.uint8), x_nib.affine)
                nib.save(p_nib, save_path+"_"+suffix)

        return imgo, (mask_mode1,)