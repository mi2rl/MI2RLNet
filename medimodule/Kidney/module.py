import os
import sys
sys.path.append('../')

from base import BaseModule
from utils import Checker

import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1.keras import backend
# from medimodule.Kidney.kidney_tumor_segmentation.load_model import kidney_tumor_segmentation
from kidney_tumor_segmentation.load_model import kidney_tumor_segmentation
from kidney_tumor_segmentation.load_model import MyModel
from kidney_tumor_segmentation.utils.load_data import Preprocessing
import SimpleITK as sitk
from kidney_tumor_segmentation.utils.run_eval_cascaded import TransAxis, resample_img_asdim, normalize_vol, CCL_check_1ststg, CCL_1ststg_post, CCL




def get_config(mode):
    config = {
        "1": { # 1st cascade
            'checkpoint': '../checkpoint/model0.h5',
            'depth': 3,
            'wlower': -300,
            'wupper': 600,
            'input_dim': (200, 200, 200),
            'num_labels_1ststg': 1
            }, 
        "2_1": {
            'checkpoint': '../checkpoint/model1.h5',
            'depth': 3,
            'wlower': -300,
            'wupper': 600,
            'input_dim': (200, 200, 200)
            },
        "2_2": {
            'checkpoint': '../checkpoint/model2.h5',
            'lossfn': 'dice',
            'depth': 4,
            'standard': 'normal',
            'task': 'tumor',
            'wlevel': 100,
            'wwidth': 400
            },
        "2_3": {
            'checkpoint': '../checkpoint/model3.h5',
            'lossfn': 'dice',
            'depth': 3,
            'standard': 'minmax',
            'task': 'tumor1',
            'wlevel': 100,
            'wwidth': 400
            },
        "2_4": {
            'checkpoint': '../checkpoint/model4.h5',
            'lossfn': 'focaldice',
            'depth': 3,
            'standard': 'minmax',
            'task': 'tumor1',
            'wlevel': 100,
            'wwidth': 400
            },
        "2_5": {
            'checkpoint': '../checkpoint/model5.h5',
            'lossfn': 'dice',
            'depth': 3,
            'standard': 'normal',
            'task': 'tumor1',
            'wlevel': 100,
            'wwidth': 400
            }}

    return config[mode]


class KidneyTumorSegmentation(BaseModule):
    def init(self, mode):
        
        self.model = kidney_tumor_segmentation(mode)
        self.config = get_config(mode)
    
    def _preprocessing(self, mode, path):
      
        if Checker.check_input_type_bool(path, 'nii'):
            if mode == '1':
                print("model1")
#                 data = sitk.ReadImage(path)
                config = self.config
                img_ct_sag = sitk.ReadImage(path)
                img_ct_axial = TransAxis(img_ct_sag, dtype=np.int16)
                raw_ct = sitk.GetArrayFromImage(img_ct_axial)

                if int(path.split('_')[1][:5]) == 223:
                    raw_ct_original = np.array(raw_ct)
                    raw_ct = raw_ct[-180:, :, :]
                    
                raw_ct_shape = np.shape(raw_ct)
                if raw_ct_shape[0] > 200:
                    is_large_z = True
                else:
                    is_large_z = False 
                
                
                # right kidney 
                if not is_large_z:
                    raw_ct_right_shape = (raw_ct_shape[0], raw_ct_shape[1], int(raw_ct_shape[2] * 3 / 5))
                    raw_ct_right_frame_shape = (200, raw_ct_shape[1], int(raw_ct_shape[2] * 3 / 5))
                    raw_ct_right_frame = np.ones(raw_ct_right_frame_shape, dtype=np.float32) * -1024
                    z_start_dst = int((200 - raw_ct_shape[0]) / 2)
                    z_end_dst = z_start_dst + raw_ct_shape[0]
                    x_start_src = 0
                    x_end_src = int(raw_ct_shape[2] * 3 / 5)
                    raw_ct_right_frame[z_start_dst:z_end_dst, :, :] = raw_ct[:, :, x_start_src:x_end_src]
                    img_ct_right = sitk.GetImageFromArray(raw_ct_right_frame)
                    img_ct_right_rs = resample_img_asdim(img_ct_right, config['input_dim'], c_val=-1024)
                    raw_ct_right_rs = sitk.GetArrayFromImage(img_ct_right_rs)
                    raw_ct_right_rs_normed = normalize_vol(raw_ct_right_rs, norm_wind_lower=config['wlower'], norm_wind_upper=config['wupper'])
                    
                    raw_ct_right_rs_normed = np.expand_dims(raw_ct_right_rs_normed, axis=0)
                    raw_ct_right_rs_normed = np.expand_dims(raw_ct_right_rs_normed, axis=-1)
 
                    
                else:
                    raw_ct_right_shape = (raw_ct_shape[0], raw_ct_shape[1], int(raw_ct_shape[2] * 3 / 5))
                    raw_pred_right_shape = [raw_ct_shape[0], 200, 200, config['num_labels_1ststg']]
                    raw_pred_right_tmp = np.zeros(shape=raw_pred_right_shape)  # raw_ct_shape[0], 200, 200, 3
                    raw_pred_right_tmp_cnt = np.zeros(shape=raw_pred_right_shape)  # raw_ct_shape[0], 200, 200, 3

                    z_list = list(np.arange(0, raw_ct_shape[0] - 200, 100)) + [raw_ct_shape[0] - 200]
                    x_start_src = 0
                    x_end_src = int(raw_ct_shape[2] * 3 / 5)

                    for z_start in z_list:
                        raw_ct_right_frame_shape = (200, raw_ct_shape[1], int(raw_ct_shape[2] * 3 / 5))
                        raw_ct_right_frame = np.ones(raw_ct_right_frame_shape, dtype=np.float32) * -1024
                        raw_ct_right_frame[:, :, :] = raw_ct[z_start:z_start + 200, :, x_start_src:x_end_src]
                        img_ct_right = sitk.GetImageFromArray(raw_ct_right_frame)
                        img_ct_right_rs = resample_img_asdim(img_ct_right, config['input_dim'], c_val=-1024)
                        raw_ct_right_rs = sitk.GetArrayFromImage(img_ct_right_rs)
                        raw_ct_right_rs_normed = normalize_vol(raw_ct_right_rs, norm_wind_lower=config['wlower'], norm_wind_upper=config['wupper'])

                        raw_ct_right_rs_normed = np.expand_dims(raw_ct_right_rs_normed, axis=0)
                        raw_ct_right_rs_normed = np.expand_dims(raw_ct_right_rs_normed, axis=-1)
                        
                        prediction = np.squeeze(model.predict(x=raw_ct_right_rs_normed), axis=0)

                        raw_pred_right_tmp[z_start:z_start + 200, :, :, :] += prediction
                        raw_pred_right_tmp_cnt[z_start:z_start + 200, :, :, :] += 1

                    raw_pred_right_tmp[np.where(raw_pred_right_tmp_cnt > 0)] /= raw_pred_right_tmp_cnt[np.where(raw_pred_right_tmp_cnt > 0)]
                    raw_ct_right_rs_normed = raw_pred_right_tmp
            
            
                # left kidney 
                if not is_large_z:
                    print("is not large z")
                    z_start_dst = int((200 - raw_ct_shape[0]) / 2)
                    z_end_dst = z_start_dst + raw_ct_shape[0]
                    x_start_src = int(raw_ct_shape[2] * 2 / 5)
                    x_end_src = raw_ct_shape[2]
                    raw_ct_left_shape = (raw_ct_shape[0], raw_ct_shape[1], x_end_src - x_start_src)
                    raw_ct_left_frame_shape = (200, raw_ct_shape[1], x_end_src - x_start_src)
                    raw_ct_left_frame = np.ones(raw_ct_left_frame_shape, dtype=np.float32) * -1024
                    raw_ct_left_frame[z_start_dst:z_end_dst, :, :] = raw_ct[:, :, x_start_src:x_end_src]
                    raw_ct_left_frame = raw_ct_left_frame[:, :, -1::-1]
                    img_ct_left = sitk.GetImageFromArray(raw_ct_left_frame)
                    img_ct_left_rs = resample_img_asdim(img_ct_left, config['input_dim'], c_val=-1024)
                    raw_ct_left_rs = sitk.GetArrayFromImage(img_ct_left_rs)
                    raw_ct_left_rs_normed = normalize_vol(raw_ct_left_rs, norm_wind_lower=config['wlower'], norm_wind_upper=config['wupper'])

                    raw_ct_left_rs_normed = np.expand_dims(raw_ct_left_rs_normed, axis=0)
                    raw_ct_left_rs_normed = np.expand_dims(raw_ct_left_rs_normed, axis=-1)
                    
                else: 
                    print("is large z")
                    raw_ct_left_shape = (raw_ct_shape[0], raw_ct_shape[1], int(raw_ct_shape[2] * 3 / 5))

                    raw_pred_left_shape = [raw_ct_shape[0], 200, 200, config['num_labels_1ststg']]
                    raw_pred_left_tmp = np.zeros(shape=raw_pred_left_shape)  # raw_ct_shape[0], 200, 200, 3
                    raw_pred_left_tmp_cnt = np.zeros(shape=raw_pred_left_shape)  # raw_ct_shape[0], 200, 200, 3

                    z_list = list(np.arange(0, raw_ct_shape[0] - 200, 100)) + [raw_ct_shape[0] - 200]
                    x_start_src = 0
                    x_end_src = int(raw_ct_shape[2] * 3 / 5)
                    for z_start in z_list:
                        raw_ct_left_frame_shape = (200, raw_ct_shape[1], int(raw_ct_shape[2] * 3 / 5))
                        raw_ct_left_frame = np.ones(raw_ct_left_frame_shape, dtype=np.float32) * -1024
                        raw_ct_left_frame[:, :, :] = raw_ct[z_start:z_start + 200, :, -raw_ct_left_frame_shape[2]:]
                        raw_ct_left_frame = raw_ct_left_frame[:, :, -1::-1]
                        img_ct_left = sitk.GetImageFromArray(raw_ct_left_frame)
                        img_ct_left_rs = resample_img_asdim(img_ct_left, config['input_dim'], c_val=-1024)
                        raw_ct_left_rs = sitk.GetArrayFromImage(img_ct_left_rs)
                        raw_ct_left_rs_normed = normalize_vol(raw_ct_left_rs, norm_wind_lower=config['wlower'], norm_wind_upper=config['wupper'])

                        raw_ct_left_rs_normed = np.expand_dims(raw_ct_left_rs_normed, axis=0)
                        raw_ct_left_rs_normed = np.expand_dims(raw_ct_left_rs_normed, axis=-1)
                        
                        prediction = np.squeeze(model.predict(x=raw_ct_left_rs_normed), axis=0)

                        raw_pred_left_tmp[z_start:z_start + 200, :, :, :] += prediction
                        raw_pred_left_tmp_cnt[z_start:z_start + 200, :, :, :] += 1

                    raw_pred_left_tmp[np.where(raw_pred_left_tmp_cnt > 0)] /= raw_pred_left_tmp_cnt[np.where(raw_pred_left_tmp_cnt > 0)]
                    raw_ct_left_rs_normed = raw_pred_left_tmp
                    z_start_dst = x_start_src
                    z_end_dst = x_end_src
                    
                return raw_ct_shape, raw_ct_right_shape, raw_ct_left_shape, raw_ct_right_rs_normed, raw_ct_left_rs_normed, is_large_z, z_start_dst, z_end_dst
           
            else: 
                if mode == '2_1':
                    config = self.config
                    print("mode2_1 preprocessing")

                    img_ct_sag = sitk.ReadImage(path)
                    img_ct_axial = TransAxis(img_ct_sag, dtype=np.int16)
                    raw_ct = sitk.GetArrayFromImage(img_ct_axial)
            
                    if int(path.split('_')[1][:5]) == 223:
                        raw_ct_original = np.array(raw_ct)
                        raw_ct = raw_ct[-180:, :, :]

                    raw_ct_shape = np.shape(raw_ct)

                    if os.path.isfile(os.path.join('./result/1', 'prediction_'+path.split('_')[1][:5]+'.nii')):
                        img_gt_sag = sitk.ReadImage(os.path.join('./result/1', 'prediction_'+path.split('_')[1][:5]+'.nii'))
                        img_gt_axial = TransAxis(img_gt_sag, dtype=np.uint8)
                        raw_gt = sitk.GetArrayFromImage(img_gt_axial)
                        if int(path.split('_')[1][:5]) == 223:
                            raw_gt_original = np.array(raw_gt)
                            raw_gt = raw_gt[-180:, :, :]
                    else:
                        raise ValueError('No masks here. Run model_1 first.')

                    idcs_label_1 = np.where(raw_gt == 1)
                    label_1_x_pos = np.mean(idcs_label_1[2])
                    idcs_label_2 = np.where(raw_gt == 2)

                    if len(idcs_label_2[0]) > len(idcs_label_1[0]) * 0.2:
                        is_both_kidney = True
                        label_2_x_pos = np.mean(idcs_label_2[2])
                    else:
                        is_both_kidney = False
                        
                    # if it is both kidney
                    if is_both_kidney:
                        if label_1_x_pos > label_2_x_pos:
                            # swap label btw. 1 and 2
                            raw_gt[idcs_label_1] = 2
                            raw_gt[idcs_label_2] = 1
                            is_left_kidney = True
                            is_right_kidney = True
                        else:
                            is_left_kidney = True
                            is_right_kidney = True
                            
                    # if one or both not kidney
                    else: 
                        if np.min(idcs_label_1[2]) < raw_ct_shape[2] / 2:
                            raw_gt[idcs_label_1] = 1
                            raw_gt[idcs_label_2] = 0
                            is_right_kidney = True
                            is_left_kidney = False
                        else:
                            raw_gt[idcs_label_1] = 2
                            raw_gt[idcs_label_2] = 0
                            is_right_kidney = False
                            is_left_kidney = True
                            
                    # extract kidney coordinate
                    if is_right_kidney:
                        idcs_label_1 = np.where(raw_gt == 1)
                        kidney_right_start = (np.max((np.min(idcs_label_1[0] - 16), 0)),
                                            np.max((np.min(idcs_label_1[1] - 16), 0)),
                                            np.max((np.min(idcs_label_1[2] - 16), 0)))
                        kidney_right_end = (np.min((np.max(idcs_label_1[0] + 16), raw_ct_shape[0])),
                                            np.min((np.max(idcs_label_1[1] + 16), raw_ct_shape[1])),
                                            np.min((np.max(idcs_label_1[2] + 16), raw_ct_shape[2])))

                    if is_left_kidney:
                        idcs_label_2 = np.where(raw_gt == 2)
                        kidney_left_start = (np.max((np.min(idcs_label_2[0] - 16), 0)),
                                            np.max((np.min(idcs_label_2[1] - 16), 0)),
                                            np.max((np.min(idcs_label_2[2] - 16), 0)))
                        kidney_left_end = (np.min((np.max(idcs_label_2[0] + 16), raw_ct_shape[0])),
                                        np.min((np.max(idcs_label_2[1] + 16), raw_ct_shape[1])),
                                        np.min((np.max(idcs_label_2[2] + 16), raw_ct_shape[2])))
                    
                    if is_right_kidney:
                        # right kidney
                        raw_ct_right_2nd_shape = (
                            int(kidney_right_end[0] - kidney_right_start[0]),
                            int(kidney_right_end[1] - kidney_right_start[1]),
                            int(kidney_right_end[2] - kidney_right_start[2]))

                        raw_ct_right_frame = np.ones(raw_ct_right_2nd_shape, dtype=np.float32) * -1024
                        raw_ct_right_frame[:, :, :] = raw_ct[kidney_right_start[0]:kidney_right_end[0],
                                                    kidney_right_start[1]:kidney_right_end[1],
                                                    kidney_right_start[2]:kidney_right_end[2]]
                        img_ct_right = sitk.GetImageFromArray(raw_ct_right_frame)
                        img_ct_right_rs = resample_img_asdim(img_ct_right, config['input_dim'], c_val=-1024)
                        raw_ct_right_rs = sitk.GetArrayFromImage(img_ct_right_rs)
                        raw_ct_right_rs_normed = normalize_vol(raw_ct_right_rs, norm_wind_lower=config['wlower'], norm_wind_upper=config['wupper'])

                        raw_ct_right_rs_normed = np.expand_dims(raw_ct_right_rs_normed, axis=0)
                        raw_ct_right_rs_normed = np.expand_dims(raw_ct_right_rs_normed, axis=-1)
                        
                    if is_left_kidney:
                        # left kidney
                        raw_ct_left_2nd_shape = (
                            int(kidney_left_end[0] - kidney_left_start[0]),
                            int(kidney_left_end[1] - kidney_left_start[1]),
                            int(kidney_left_end[2] - kidney_left_start[2]))
                        raw_ct_left_frame = np.ones(raw_ct_left_2nd_shape, dtype=np.float32) * -1024
                        raw_ct_left_frame[:, :, :] = raw_ct[kidney_left_start[0]:kidney_left_end[0],
                                                    kidney_left_start[1]:kidney_left_end[1],
                                                    kidney_left_start[2]:kidney_left_end[2]]
                        raw_ct_left_frame = raw_ct_left_frame[:, :, -1::-1]
                        img_ct_left = sitk.GetImageFromArray(raw_ct_left_frame)
                        img_ct_left_rs = resample_img_asdim(img_ct_left, config['input_dim'], c_val=-1024)
                        raw_ct_left_rs = sitk.GetArrayFromImage(img_ct_left_rs)
                        raw_ct_left_rs_normed = normalize_vol(raw_ct_left_rs, norm_wind_lower=config['wlower'], norm_wind_upper=config['wupper'])

                        raw_ct_left_rs_normed = np.expand_dims(raw_ct_left_rs_normed, axis=0)
                        raw_ct_left_rs_normed = np.expand_dims(raw_ct_left_rs_normed, axis=-1)

                    return raw_ct_right_rs_normed, raw_ct_left_rs_normed, raw_ct_right_2nd_shape, raw_ct_left_2nd_shape, raw_ct, is_right_kidney, is_left_kidney, kidney_right_start, kidney_right_end, kidney_left_start, kidney_left_end

                else:
                    config = self.config
                    print("mode2_5")
                 
                    prep = Preprocessing(
                        task=config['task'],
                        standard=config['standard'],
                        wlevel=config['wlevel'],
                        wwidth=config['wwidth'],
                        rotation_range=[0., 0., 0.]
                        )
                    img_orig = sitk.ReadImage(path)
                    mask_orig = sitk.ReadImage(os.path.join('./result/1', 'prediction_'+path.split('_')[1][:5]+'.nii'))
                    
                    return img_orig, mask_orig, prep
                             
        else:
            raise ValueError('Please check the sample in README.md.')
            

    #### Prediction ####
    def predict(self, mode, path):
        if mode == '1':
            config = self.config
            model = self.model
            print("mode1 prediction")
            raw_ct_shape, raw_ct_right_shape, raw_ct_left_shape, raw_ct_right_rs_normed, raw_ct_left_rs_normed, is_large_z, z_start_dst, z_end_dst  = self._preprocessing(mode, path)
            
            # right
            if not is_large_z:
                prediction = model.predict(x=raw_ct_right_rs_normed)
                if np.shape(prediction)[-1] == 1:
                    prediction = np.squeeze(prediction)
                else:
                    prediction = np.squeeze(np.argmax(prediction, axis=-1))

                prediction = prediction[z_start_dst:z_end_dst, :, :]

                raw_pred_right = sitk.GetArrayFromImage(
                    resample_img_asdim(sitk.GetImageFromArray(prediction), tuple(reversed(raw_ct_right_shape)), interp=sitk.sitkNearestNeighbor))
                raw_pred_right[np.where(raw_pred_right > 0.5)] = 1
                raw_pred_right = CCL_check_1ststg(raw_pred_right)
            
            else:
                if config['num_labels_1ststg'] != 1:
                    prediction = np.argmax(raw_pred_right_tmp, axis=-1)
                else:
                    prediction = np.squeeze(raw_pred_right_tmp)
                    prediction[np.where(prediction > 0.5)] = 1

                raw_pred_right = sitk.GetArrayFromImage(
                    resample_img_asdim(sitk.GetImageFromArray(prediction), tuple(reversed(raw_ct_right_shape)), interp=sitk.sitkNearestNeighbor))
                raw_pred_right[np.where(raw_pred_right > 0.5)] = 1
                raw_pred_right = CCL_check_1ststg(raw_pred_right)
            
            
            # left
            if not is_large_z:
                prediction = model.predict(x=raw_ct_left_rs_normed)

                if np.shape(prediction)[-1] == 1:
                    prediction = np.squeeze(prediction)
                else:
                    prediction = np.squeeze(np.argmax(prediction, axis=-1))

                prediction = prediction[z_start_dst:z_end_dst, :, :]

                raw_pred_left = sitk.GetArrayFromImage(
                    resample_img_asdim(sitk.GetImageFromArray(prediction), tuple(reversed(raw_ct_left_shape)), interp=sitk.sitkNearestNeighbor))
                raw_pred_left[np.where(raw_pred_left > 0.5)] = 1
                raw_pred_left = CCL_check_1ststg(raw_pred_left)

            
            else:
                
                if config['num_labels_1ststg'] != 1:
                    prediction = np.argmax(raw_pred_left_tmp, axis=-1)
                else:
                    prediction = np.squeeze(raw_pred_left_tmp)
                    prediction[np.where(prediction > 0.5)] = 1

                raw_pred_left = sitk.GetArrayFromImage(
                    resample_img_asdim(sitk.GetImageFromArray(prediction), tuple(reversed(raw_ct_left_shape)), interp=sitk.sitkNearestNeighbor))
                raw_pred_left[np.where(raw_pred_left > 0.5)] = 1
                raw_pred_left = CCL_check_1ststg(raw_pred_left)
                
            # check if both kidneys are valid
            raw_pred_whole = np.zeros(raw_ct_shape, dtype=np.uint8)
            raw_pred_right_shape = np.shape(raw_pred_right)
            raw_pred_whole[:, :, :raw_pred_right_shape[2]] = raw_pred_right
            raw_pred_left_shape = np.shape(raw_pred_left)
            raw_pred_left[:, :, :] = raw_pred_left[:, :, -1::-1]
            raw_pred_whole_left_tmp = raw_pred_whole[:, :, -raw_pred_left_shape[2]:]
            raw_pred_whole_left_tmp[np.where(raw_pred_left > 0)] = raw_pred_left[np.where(raw_pred_left > 0)]
            raw_pred_whole[:, :, -raw_pred_left_shape[2]:] = raw_pred_whole_left_tmp
            raw_pred_whole = CCL_1ststg_post(raw_pred_whole)

            if int(path.split('_')[1][:5]) == 223:
                raw_pred_whole_tmp = np.zeros(np.shape(raw_ct_original), dtype=np.uint8)
                raw_pred_whole_tmp[-180:, :, :] = raw_pred_whole
                raw_pred_whole = raw_pred_whole_tmp    
           
            return raw_pred_whole
        
        else: 
            if mode == '2_1':
                config = self.config
                model = self.model
                raw_ct_right_rs_normed, raw_ct_left_rs_normed, raw_ct_right_2nd_shape, raw_ct_left_2nd_shape, raw_ct, is_right_kidney, is_left_kidney, kidney_right_start, kidney_right_end, kidney_left_start, kidney_left_end = self._preprocessing(mode, path)
#                 raw_ct_shape, raw_ct_right_shape, raw_ct_left_shape, raw_ct_right_rs_normed, raw_ct_left_rs_normed, is_large_z, z_start_dst, z_end_dst
                print("model2_1 prediction")
                if is_right_kidney:
                    prediction = model.predict(x=raw_ct_right_rs_normed)
                    if np.shape(prediction)[-1] == 1:
                        prediction = np.squeeze(prediction)
                    else:
                        prediction = np.squeeze(np.argmax(prediction, axis=-1))

                    raw_pred_right = sitk.GetArrayFromImage(
                        resample_img_asdim(sitk.GetImageFromArray(prediction), tuple(reversed(raw_ct_right_2nd_shape)), interp=sitk.sitkNearestNeighbor))

                    raw_pred_right_tmp = np.array(raw_pred_right)
                    raw_pred_right_tmp[np.where(raw_pred_right_tmp > 0)] = 1
                    raw_pred_right_tmp = CCL(raw_pred_right_tmp, num_labels=2)
                    raw_pred_right[np.where(raw_pred_right_tmp == 0)] = 0
                    raw_ct_right = np.array(raw_ct[kidney_right_start[0]:kidney_right_end[0],
                                            kidney_right_start[1]:kidney_right_end[1],
                                            kidney_right_start[2]:kidney_right_end[2]])
                if is_left_kidney:
                    prediction = model.predict(x=raw_ct_left_rs_normed)

                    if np.shape(prediction)[-1] == 1:
                        prediction = np.squeeze(prediction)
                    else:
                        prediction = np.squeeze(np.argmax(prediction, axis=-1))

                    raw_pred_left = sitk.GetArrayFromImage(
                        resample_img_asdim(sitk.GetImageFromArray(prediction), tuple(reversed(raw_ct_left_2nd_shape)), interp=sitk.sitkNearestNeighbor))
                    raw_pred_left = raw_pred_left[:, :, -1::-1]

                    raw_pred_left_tmp = np.array(raw_pred_left)
                    raw_pred_left_tmp[np.where(raw_pred_left_tmp > 0)] = 1
                    raw_pred_left_tmp = CCL(raw_pred_left_tmp, num_labels=2)
                    raw_pred_left[np.where(raw_pred_left_tmp == 0)] = 0
                    raw_ct_left = np.array(raw_ct[kidney_left_start[0]:kidney_left_end[0],
                                        kidney_left_start[1]:kidney_left_end[1],
                                        kidney_left_start[2]:kidney_left_end[2]])
                    
                raw_pred_whole = np.zeros(np.shape(raw_ct), dtype=np.uint8)
                
                if is_right_kidney:
                    raw_pred_whole[kidney_right_start[0]:kidney_right_end[0], kidney_right_start[1]:kidney_right_end[1],
                    kidney_right_start[2]:kidney_right_end[2]] = raw_pred_right
                if is_left_kidney:
                    raw_pred_whole_left_tmp = raw_pred_whole[kidney_left_start[0]:kidney_left_end[0],
                                            kidney_left_start[1]:kidney_left_end[1], kidney_left_start[2]:kidney_left_end[2]]
                    raw_pred_whole_left_tmp[np.where(raw_pred_left > 0)] = raw_pred_left[np.where(raw_pred_left > 0)]
                    raw_pred_whole[kidney_left_start[0]:kidney_left_end[0], kidney_left_start[1]:kidney_left_end[1],
                    kidney_left_start[2]:kidney_left_end[2]] = raw_pred_whole_left_tmp

                if int(path.split('_')[1][:5]) == 223:
                    raw_pred_whole_tmp = np.zeros(np.shape(raw_ct_original), dtype=np.uint8)
                    raw_pred_whole_tmp[-180:, :, :] = raw_pred_whole
                    raw_pred_whole = raw_pred_whole_tmp
                    
                return raw_pred_whole
            
            else: 
                model = self.model
                config = self.config
                print("mode2_5")
                img_orig, mask_orig, prep = self._preprocessing(mode, path)
                
                result_save = np.zeros_like(sitk.GetArrayFromImage(mask_orig))
                
                loop = 2 if config['task'] == 'tumor' else 1
                
                for idx in range(loop):
                    img, mask, spacing = prep._array2img([img_orig, mask_orig], True)
                    if config['task'] == 'tumor':
                        img, mask, flag, bbox = prep._getvoi([img, mask, idx], True)
                    else:
                        img, mask, flag, bbox, diff, diff1 = prep._getvoi([img, mask, idx], True)
                    if flag:
                        if idx == 1 and config['task'] == 'tumor':
                            img, mask = prep._horizontal_flip([img, mask])
                            
                        img = prep._windowing(img)
                        img = prep._standard(img)
                        mask = prep._onehot(mask)
                        img, mask = prep._expand([img, mask])
                
                        result = model.mymodel.predict_on_batch(img)
                        result = np.argmax(np.squeeze(result), axis=-1)
                        label = np.argmax(np.squeeze(mask), axis=-1)


                        if config['task'] == 'tumor':
                            if idx == 1:
                                img, result = prep._horizontal_flip([img, result])
                            result_save[np.maximum(0, bbox[0]):np.minimum(result_save.shape[0]-1, bbox[1]+1),
                                        np.maximum(0, bbox[2]):np.minimum(result_save.shape[1]-1, bbox[3]+1),
                                        np.maximum(0, bbox[4]):np.minimum(result_save.shape[2]-1, bbox[5]+1)] = result
                        
                        elif config['task'] == 'tumor1':
                            threshold = [380, 230, 72]
                            mask_orig = sitk.GetArrayFromImage(mask_orig)
                            result_save[np.maximum(0,bbox[0]):np.minimum(result_save.shape[0],bbox[1]),
                                        np.maximum(0,bbox[2]):np.minimum(result_save.shape[1],bbox[3]),
                                        np.maximum(0,bbox[4]):np.minimum(result_save.shape[2],bbox[5])] = result[diff[0]//2:-diff[0]//2-diff1[0] if -diff[0]//2-diff1[0] != 0 else result.shape[0],
                                                                                                                 diff[1]//2:-diff[1]//2-diff1[1] if -diff[1]//2-diff1[1] != 0 else result.shape[1],
                                                                                                                 diff[2]//2:-diff[2]//2-diff1[2] if -diff[2]//2-diff1[2] != 0 else result.shape[2]]
                        
                temp2 = np.swapaxes(result_save, 1, 2)
                temp2 = np.swapaxes(temp2, 0, 1)
                temp2 = np.swapaxes(temp2, 1, 2)
                
                return temp2
    
if __name__ == "__main__":
    
    Checker.set_gpu('4', 'tf2')
#     backend.set_session(get_session())
    tumor_segmentation = KidneyTumorSegmentation()
    tumor_segmentation.init('2_3')
#     img_rt , img_lt, is_large_z = tumor_segmentation._preprocessing("1", '../../data/interpolated/case_00000/imaging.nii')
#     img_rt , img_lt, is_right_kidney, is_left_kidney = tumor_segmentation._preprocessing("1", '../../data/interpolated/case_00000/imaging.nii')
#     img_orig, mask_orig, prep = tumor_segmentation._preprocessing("2_3", '../../data/interpolated/case_00000/imaging.nii')
#     print(img_rt.shape, img_lt.shape)
    raw_pred_whole = tumor_segmentation.predict("2_3", '../../data/interpolated/case_00000/imaging.nii')
    print(raw_pred_whole.shape)
    