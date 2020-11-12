import numpy as np
import tensorflow as tf
import csv
# import queue
import os

from skimage import color

os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % 0
from pathlib import Path
# import itertools
import SimpleITK as sitk
# from .util_args import args
# from .mylogger import eval_logger
# import glob
import scipy.ndimage.morphology
# import time
import nibabel as nib

'''
Kidney(1): Red
Cancer(2): Magenta
'''


def subtract_erode_img(src):
    src_shape = src.shape

    ret_img = np.zeros(src_shape)
    for z in range(src_shape[0]):
        tmp_img_2d = src[z, :, :]
        if np.sum(tmp_img_2d) == 0:
            continue
        tmp_img_2d_eroded = scipy.ndimage.morphology.binary_erosion(tmp_img_2d)
        ret_img[z, :, :] = tmp_img_2d - tmp_img_2d_eroded

    # eroded_img = scipy.ndimage.morphology.binary_erosion(src)
    # ret_img = src - eroded_img
    return ret_img


def load_GT_mask(dir_GT, pid, rtlt, shape):
    fn_gt = os.path.join(dir_GT, 'case_%04d_%04d_GT_200x200x200.raw' % (pid, rtlt))

    raw_gt = np.fromfile(fn_gt, dtype=np.uint8, count=np.product(shape)).reshape(shape)

    return raw_gt


def colorize_boundary_vol(CT_raw, seg_raw, norm_wind_lower, norm_wind_upper, fn_out_fullpath=None):
    '''
    Kidney(1): Red
    Cancer(2): Magenta
    '''

    CT_raw_r = (CT_raw - norm_wind_lower) / (norm_wind_upper - norm_wind_lower)
    CT_raw_r[np.where(CT_raw_r < 0)] = 0
    CT_raw_r[np.where(CT_raw_r > 1)] = 1
    CT_raw_r = np.array(CT_raw_r)

    CT_raw_shape = np.shape(CT_raw)
    CT_raw_rgb = np.stack((CT_raw_r, CT_raw_r, CT_raw_r), axis=3)
    total_mask_rgb = np.zeros(np.shape(CT_raw_rgb))

    empty = np.zeros(shape=np.shape(CT_raw_r))

    for idx in range(1, 3):
        print('idx: %d' % idx)
        seg_tmp = np.zeros(shape=CT_raw_shape)
        seg_tmp[np.where(seg_raw == idx)] = 1
        tmp = subtract_erode_img(seg_tmp)
        CT_raw_rgb_r = CT_raw_rgb[:, :, :, 0]
        CT_raw_rgb_g = CT_raw_rgb[:, :, :, 1]
        CT_raw_rgb_b = CT_raw_rgb[:, :, :, 2]
        if idx == 1:  # Kidney (Red)
            tmp_mask_rgb = np.stack((tmp, empty, empty), axis=3)
            CT_raw_rgb_r[np.where(tmp == 1)] = 1
            CT_raw_rgb_g[np.where(tmp == 1)] = 0
            CT_raw_rgb_b[np.where(tmp == 1)] = 0
        elif idx == 2:  # Cancer (Magenta)
            tmp_mask_rgb = np.stack((tmp, empty, tmp), axis=3)
            CT_raw_rgb_r[np.where(tmp == 1)] = 1
            CT_raw_rgb_g[np.where(tmp == 1)] = 0
            CT_raw_rgb_b[np.where(tmp == 1)] = 1

        CT_raw_rgb[:, :, :, 0] = CT_raw_rgb_r
        CT_raw_rgb[:, :, :, 1] = CT_raw_rgb_g
        CT_raw_rgb[:, :, :, 2] = CT_raw_rgb_b

        total_mask_rgb[np.where(tmp_mask_rgb > 0)] = tmp_mask_rgb[np.where(tmp_mask_rgb > 0)]

    # CT_raw_rgb[np.where(total_mask_rgb > 0)] = total_mask_rgb[np.where(total_mask_rgb > 0)]

    if fn_out_fullpath is not None:
        np.array(CT_raw_rgb * 255).astype(np.uint8).tofile(fn_out_fullpath)

    return np.array(CT_raw_rgb * 255).astype(np.uint8)


def colorize_vol(CT_raw, seg_raw, norm_wind_lower, norm_wind_upper, fn_out_fullpath=None):
    alpha = 0.8
    CT_raw_r = (CT_raw - norm_wind_lower) / (norm_wind_upper - norm_wind_lower)
    CT_raw_r[np.where(CT_raw_r < 0)] = 0
    CT_raw_r[np.where(CT_raw_r > 1)] = 1
    CT_raw_r = np.array(CT_raw_r)

    CT_raw_shape = np.shape(CT_raw)

    # for z in range(CT_raw_shape[0]):

    empty_mask = np.zeros(shape=np.shape(CT_raw_r))

    CT_raw_rgb = np.stack((CT_raw_r, CT_raw_r, CT_raw_r), axis=3)
    CT_raw_rgb_tmp = np.array(CT_raw_rgb)
    CT_raw_rgb_tmp[np.where(CT_raw_rgb_tmp < 0.8)] = 0.8
    total_mask_rgb = np.zeros(np.shape(CT_raw_rgb))
    for idx in range(1, 5):
        tmp_mask = np.zeros(shape=np.shape(CT_raw_r))
        tmp_mask[np.where(seg_raw == idx)] = 1
        if idx == 1:  # Kidney
            tmp_mask_rgb = np.stack((tmp_mask, empty_mask, empty_mask), axis=3)
        elif idx == 2:  # Cancer
            tmp_mask_rgb = np.stack((tmp_mask, empty_mask, tmp_mask), axis=3)
        total_mask_rgb[np.where(tmp_mask_rgb > 0)] = tmp_mask_rgb[np.where(tmp_mask_rgb > 0)]

    # total_mask_rgb.tofile('\\\\data5\\Workspace\\yongjin\\Challenge_SegTHOR\\eval\\ckpt-66-valDice_0_0.985-valDice_1_0.581-valDice_2_0.872-valDice_3_0.680-valDice_4_0.728.h5\\Patient_01\\total_mask.raw')

    # CT_raw_rgb.tofile('\\\\data5\\Workspace\\yongjin\\Challenge_SegTHOR\\eval\\ckpt-66-valDice_0_0.985-valDice_1_0.581-valDice_2_0.872-valDice_3_0.680-valDice_4_0.728.h5\\Patient_01\\CT_raw_rgb.raw')
    img_masked = np.zeros(shape=np.shape(CT_raw_rgb))
    print('CT_raw_rgb shape: ' + str(np.shape(CT_raw_rgb)))

    for z in range(CT_raw_shape[0]):
        CT_raw_hsv = color.rgb2hsv(np.squeeze(CT_raw_rgb[z, :, :, :]))
        color_mask_hsv = color.rgb2hsv(np.squeeze(total_mask_rgb[z, :, :, :]))

        CT_raw_hsv[..., 0] = color_mask_hsv[..., 0]
        CT_raw_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

        img_masked_tmp = color.hsv2rgb(CT_raw_hsv)
        img_masked[z, :, :, :] = img_masked_tmp

    if fn_out_fullpath is not None:
        np.array(img_masked * 255).astype(np.uint8).tofile(fn_out_fullpath)

    return np.array(img_masked * 255).astype(np.uint8)


def colorize_boundary_and_fill_vol(CT_raw, seg_raw, norm_wind_lower, norm_wind_upper, fn_out_fullpath=None):
    '''
    Kidney(1)
    Cancer(2)
    '''

    alpha = 0.5
    CT_raw_r = (CT_raw - norm_wind_lower) / (norm_wind_upper - norm_wind_lower)
    CT_raw_r[np.where(CT_raw_r < 0)] = 0
    CT_raw_r[np.where(CT_raw_r > 1)] = 1
    CT_raw_r *= 255
    CT_raw_r = np.array(CT_raw_r).astype(np.uint8)

    CT_raw_shape = np.shape(CT_raw)

    # for z in range(CT_raw_shape[0]):

    empty_mask = np.zeros(shape=np.shape(CT_raw_r))

    CT_raw_rgb = np.stack((CT_raw_r, CT_raw_r, CT_raw_r), axis=3).astype(np.uint8)
    # CT_raw_rgb_tmp = np.array(CT_raw_rgb, dtype=np.uint8)
    # CT_raw_rgb_tmp[np.where(CT_raw_rgb_tmp < alpha)] = alpha
    total_mask_rgb = np.zeros(np.shape(CT_raw_rgb))
    CT_raw_rgb_r = CT_raw_rgb[:, :, :, 0]
    CT_raw_rgb_g = CT_raw_rgb[:, :, :, 1]
    CT_raw_rgb_b = CT_raw_rgb[:, :, :, 2]
    for idx in range(1, 3):
        tmp_mask = np.zeros(shape=np.shape(CT_raw_r))
        tmp_mask[np.where(seg_raw == idx)] = 1
        tmp = tmp_mask
        tmp_b = subtract_erode_img(tmp_mask)
        tmp_mask_rgb = np.zeros(np.shape(CT_raw_rgb), dtype=np.uint8)
        tmp_mask_r = tmp_mask_rgb[:, :, :, 0]
        tmp_mask_g = tmp_mask_rgb[:, :, :, 1]
        tmp_mask_b = tmp_mask_rgb[:, :, :, 2]

        if idx == 1:  # Kidney  (128, 0, 0)
            tmp_mask_r[np.where(tmp == 1)] = 128
            tmp_mask_g[np.where(tmp == 1)] = 0
            tmp_mask_b[np.where(tmp == 1)] = 0
            CT_raw_rgb_r[np.where(tmp_b == 1)] = 128
            CT_raw_rgb_g[np.where(tmp_b == 1)] = 0
            CT_raw_rgb_b[np.where(tmp_b == 1)] = 0
        elif idx == 2:  # Cancer (153, 99, 36)
            tmp_mask_r[np.where(tmp == 1)] = 60
            tmp_mask_g[np.where(tmp == 1)] = 180
            tmp_mask_b[np.where(tmp == 1)] = 75
            CT_raw_rgb_r[np.where(tmp_b == 1)] = 60
            CT_raw_rgb_g[np.where(tmp_b == 1)] = 180
            CT_raw_rgb_b[np.where(tmp_b == 1)] = 75

        # tmp_mask_r /= 255
        # tmp_mask_g /= 255
        # tmp_mask_b /= 255
        tmp_mask_rgb = np.stack((tmp_mask_r, tmp_mask_g, tmp_mask_b), axis=3).astype(np.float32) / 255

        total_mask_rgb[np.where(tmp_mask_rgb > 0)] = tmp_mask_rgb[np.where(tmp_mask_rgb > 0)]

    # CT_raw_rgb_r /= 255
    # CT_raw_rgb_g /= 255
    # CT_raw_rgb_b /= 255
    CT_raw_rgb = np.stack((CT_raw_rgb_r, CT_raw_rgb_g, CT_raw_rgb_b), axis=3).astype(np.float32) / 255

    # total_mask_rgb.tofile('\\\\data5\\Workspace\\yongjin\\Challenge_SegTHOR\\eval\\ckpt-66-valDice_0_0.985-valDice_1_0.581-valDice_2_0.872-valDice_3_0.680-valDice_4_0.728.h5\\Patient_01\\total_mask.raw')

    # CT_raw_rgb.tofile('\\\\data5\\Workspace\\yongjin\\Challenge_SegTHOR\\eval\\ckpt-66-valDice_0_0.985-valDice_1_0.581-valDice_2_0.872-valDice_3_0.680-valDice_4_0.728.h5\\Patient_01\\CT_raw_rgb.raw')
    img_masked = np.zeros(shape=np.shape(CT_raw_rgb))
    print('CT_raw_rgb shape: ' + str(np.shape(CT_raw_rgb)))

    for z in range(CT_raw_shape[0]):
        CT_raw_hsv = color.rgb2hsv(np.squeeze(CT_raw_rgb[z, :, :, :]))
        color_mask_hsv = color.rgb2hsv(np.squeeze(total_mask_rgb[z, :, :, :]))

        CT_raw_hsv[..., 0] = color_mask_hsv[..., 0]
        CT_raw_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

        img_masked_tmp = color.hsv2rgb(CT_raw_hsv)
        img_masked[z, :, :, :] = img_masked_tmp

    if fn_out_fullpath is not None:
        np.array(img_masked * 255).astype(np.uint8).tofile(fn_out_fullpath)

    return np.array(img_masked * 255).astype(np.uint8)


def GetDice(GT_raw, pred_raw, fn_csv, num_label=1):
    GT_shape = np.shape(GT_raw)
    dice_list = []

    num_label_tmp = 2 if num_label == 1 else num_label

    real_num_label = 1

    for label in range(1, num_label_tmp):
        GT_raw_label = np.zeros(GT_shape)
        pred_raw_label = np.zeros(GT_shape)
        if label == 1:
            GT_raw_label[np.where(GT_raw > 0)] = 1
            pred_raw_label[np.where(pred_raw > 0)] = 1
        else:
            GT_raw_label[np.where(GT_raw == label)] = 1
            pred_raw_label[np.where(pred_raw == label)] = 1

        idcs = np.where(GT_raw == label)
        if len(idcs[0]) == 0:
            continue

        intersect = GT_raw_label * pred_raw_label
        dice = 2 * np.sum(intersect) / (np.sum(GT_raw_label) + np.sum(pred_raw_label) + 1e-7)
        dice_list.append(dice)
        real_num_label += 1

    with open(fn_csv, 'w', newline='') as fid:
        writer = csv.writer(fid)
        for idx in range(1, real_num_label):
            writer.writerow([idx, dice_list[idx - 1]])

    return dice_list


def main():
    fn_model_1st = '\\\\data5\\Workspace\\yongjin\\KidneySeg\\Model\\ACE_C_v1_seed_1000\\ckpt-240-valDice_0.939.h5'
    fn_model_2nd = '\\\\data5\\Workspace\\yongjin\\KidneySeg\\Model_2ndstg\\ACE_C_2ndstg_v1_seed_1108\\ckpt-490-valAvgDice_0.839-valDice0_0.994-valDice1_0.874-valDice2_0.650.h5'
    dir_src_base='F:\\ImageData\\kits19_interp\\data'
    dir_output_base = '\\\\data5\\Workspace\\yongjin\\KidneySeg\\eval'

    net_input_dim = (200, 200, 200)
    num_channels = 1
    zdim, ydim, xdim = net_input_dim
    config = tf.compat.v1.ConfigProto()  # tensorflow2.3.0
    config.gpu_options.allow_growth = True  # Need to check

    '''                             Load 1st stage model                                            '''
    #                               SE-Unet w/ last relu w/ 1 label                             #
    num_labels_1ststg = 1
    from Model_ACE_CNet import load_model as load_model_1st
    model_1st = load_model_1st(input_shape=(zdim, ydim, xdim, num_channels), num_labels=1, base_filter=32,
                               depth_size=3, se_res_block=True, se_ratio=16, last_relu=True)
    model_1st.load_weights(fn_model_1st)

    '''                             Load 2nd stage model                                            '''
    #                               SE-Unet                             #
    from cascade_2nd.model_1.Model_ACE_CNet_2ndstg import load_model as load_model_2nd
    model_2nd = load_model_2nd(input_shape=(zdim, ydim, xdim, num_channels), num_labels=3, base_filter=32,
                               depth_size=3, se_res_block=True, se_ratio=16, last_relu=False)
    model_2nd.load_weights(fn_model_2nd)

    dir_output = fn_model_1st.split(os.sep)[-2] + '-' + \
                 fn_model_1st.split(os.sep)[-1].split('-')[0] + '-' + fn_model_1st.split(os.sep)[-1].split('-')[1] + \
                 '-' + fn_model_2nd.split(os.sep)[-2] + '-' + \
                 fn_model_2nd.split(os.sep)[-1].split('-')[0] + '-' + fn_model_2nd.split(os.sep)[-1].split('-')[1]

    # pid_range = [40, 165, 194]
    # pid_range = pid_range + list(range(0, 5)) + list(range(6, 15)) + list(range(16, 20))
    # pid_range = pid_range + list(range(210, 300))  # challenge test set

    pid_range = list(range(210, 300))  # challenge test set

    for pid in pid_range:
        print('pid: %d' % pid)
        dir_src = os.path.join(dir_src_base, 'case_%05d' % pid)
        dir_output_final = os.path.join(dir_output_base, dir_output, 'case_%05d' % pid)
        Path(dir_output_final).mkdir(parents=True, exist_ok=True)
        fn_img_ct = os.path.join(dir_src, 'imaging.nii.gz')
        fn_img_gt = os.path.join(dir_src, 'segmentation.nii.gz')

        ''' load and transpose '''
        img_ct_sag = sitk.ReadImage(fn_img_ct)
        img_ct_axial = TransAxis(img_ct_sag, dtype=np.int16)
        raw_ct = sitk.GetArrayFromImage(img_ct_axial)

        # test set 223 for hard coding
        if pid == 223:
            raw_ct_original = np.array(raw_ct)
            raw_ct = raw_ct[-180:, :, :]

        if os.path.isfile(fn_img_gt):
            img_gt_sag = sitk.ReadImage(fn_img_gt)
            img_gt_axial = TransAxis(img_gt_sag, dtype=np.uint8)
            raw_gt = sitk.GetArrayFromImage(img_gt_axial)
        else:
            raw_gt = None

        raw_ct_shape = np.shape(raw_ct)
        if raw_ct_shape[0] > 200:
            is_large_z = True
        else:
            is_large_z = False

        ''' 1st stage '''
        norm_wind_lower = -300
        norm_wind_upper = 600

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
            img_ct_right_rs = resample_img_asdim(img_ct_right, net_input_dim, c_val=-1024)
            raw_ct_right_rs = sitk.GetArrayFromImage(img_ct_right_rs)
            raw_ct_right_rs_normed = normalize_vol(raw_ct_right_rs, norm_wind_lower=norm_wind_lower,
                                                   norm_wind_upper=norm_wind_upper)

            fn_raw_ct_right_rs_normed = os.path.join(dir_output_final, 'raw_right_ct_%dx%dx%d.raw' % tuple(
                reversed(np.shape(raw_ct_right_rs_normed))))
            np.array(raw_ct_right_rs_normed).astype(np.float32).tofile(fn_raw_ct_right_rs_normed)

            raw_ct_right_rs_normed = np.expand_dims(raw_ct_right_rs_normed, axis=0)
            raw_ct_right_rs_normed = np.expand_dims(raw_ct_right_rs_normed, axis=-1)
            prediction = model_1st.predict(x=raw_ct_right_rs_normed)

            fn_pred = os.path.join(dir_output_final, 'pred_right.raw')
            np.transpose(prediction, [0, 4, 1, 2, 3]).astype(np.float32).tofile(fn_pred)

            np.array(raw_ct_right_rs_normed).astype(np.float32).tofile(fn_raw_ct_right_rs_normed)
            if np.shape(prediction)[-1] == 1:
                prediction = np.squeeze(prediction)
            else:
                prediction = np.squeeze(np.argmax(prediction, axis=-1))

            prediction = prediction[z_start_dst:z_end_dst, :, :]

            raw_pred_right = sitk.GetArrayFromImage(
                resample_img_asdim(sitk.GetImageFromArray(prediction), tuple(reversed(raw_ct_right_shape)),
                                   interp=sitk.sitkNearestNeighbor))
            raw_pred_right[np.where(raw_pred_right > 0.5)] = 1
            print('1st right')
            raw_pred_right = CCL_check_1ststg(raw_pred_right)

        else:
            raw_ct_right_shape = (raw_ct_shape[0], raw_ct_shape[1], int(raw_ct_shape[2] * 3 / 5))

            raw_pred_right_shape = [raw_ct_shape[0], 200, 200, num_labels_1ststg]
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
                img_ct_right_rs = resample_img_asdim(img_ct_right, net_input_dim, c_val=-1024)
                raw_ct_right_rs = sitk.GetArrayFromImage(img_ct_right_rs)
                raw_ct_right_rs_normed = normalize_vol(raw_ct_right_rs, norm_wind_lower=norm_wind_lower,
                                                       norm_wind_upper=norm_wind_upper)

                fn_raw_ct_right_rs_normed = os.path.join(dir_output_final, 'raw_right_ct_%dx%dx%d.raw' % tuple(
                    reversed(np.shape(raw_ct_right_rs_normed))))
                np.array(raw_ct_right_rs_normed).astype(np.float32).tofile(fn_raw_ct_right_rs_normed)

                raw_ct_right_rs_normed = np.expand_dims(raw_ct_right_rs_normed, axis=0)
                raw_ct_right_rs_normed = np.expand_dims(raw_ct_right_rs_normed, axis=-1)
                prediction = np.squeeze(model_1st.predict(x=raw_ct_right_rs_normed), axis=0)

                raw_pred_right_tmp[z_start:z_start + 200, :, :, :] += prediction
                raw_pred_right_tmp_cnt[z_start:z_start + 200, :, :, :] += 1

            raw_pred_right_tmp[np.where(raw_pred_right_tmp_cnt > 0)] /= raw_pred_right_tmp_cnt[
                np.where(raw_pred_right_tmp_cnt > 0)]

            fn_pred = os.path.join(dir_output_final, 'pred_right.raw')
            np.transpose(raw_pred_right_tmp, [3, 0, 1, 2]).astype(np.float32).tofile(fn_pred)

            if num_labels_1ststg != 1:
                prediction = np.argmax(raw_pred_right_tmp, axis=-1)
            else:
                prediction = np.squeeze(raw_pred_right_tmp)
                prediction[np.where(prediction > 0.5)] = 1

            raw_pred_right = sitk.GetArrayFromImage(
                resample_img_asdim(sitk.GetImageFromArray(prediction), tuple(reversed(raw_ct_right_shape)),
                                   interp=sitk.sitkNearestNeighbor))
            raw_pred_right[np.where(raw_pred_right > 0.5)] = 1
            print('1st right')
            raw_pred_right = CCL_check_1ststg(raw_pred_right)

        # left kidney
        if not is_large_z:
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
            img_ct_left_rs = resample_img_asdim(img_ct_left, net_input_dim, c_val=-1024)
            raw_ct_left_rs = sitk.GetArrayFromImage(img_ct_left_rs)
            raw_ct_left_rs_normed = normalize_vol(raw_ct_left_rs, norm_wind_lower=norm_wind_lower,
                                                  norm_wind_upper=norm_wind_upper)

            fn_raw_ct_left_rs_normed = os.path.join(dir_output_final, 'raw_left_ct_%dx%dx%d.raw' % tuple(
                reversed(np.shape(raw_ct_left_rs_normed))))
            np.array(raw_ct_left_rs_normed).astype(np.float32).tofile(fn_raw_ct_left_rs_normed)

            raw_ct_left_rs_normed = np.expand_dims(raw_ct_left_rs_normed, axis=0)
            raw_ct_left_rs_normed = np.expand_dims(raw_ct_left_rs_normed, axis=-1)
            prediction = model_1st.predict(x=raw_ct_left_rs_normed)

            fn_pred = os.path.join(dir_output_final, 'pred_left.raw')
            np.transpose(prediction, [0, 4, 1, 2, 3]).astype(np.float32).tofile(fn_pred)

            np.array(raw_ct_left_rs_normed).astype(np.float32).tofile(fn_raw_ct_left_rs_normed)
            if np.shape(prediction)[-1] == 1:
                prediction = np.squeeze(prediction)
            else:
                prediction = np.squeeze(np.argmax(prediction, axis=-1))

            prediction = prediction[z_start_dst:z_end_dst, :, :]

            raw_pred_left = sitk.GetArrayFromImage(
                resample_img_asdim(sitk.GetImageFromArray(prediction), tuple(reversed(raw_ct_left_shape)),
                                   interp=sitk.sitkNearestNeighbor))
            raw_pred_left[np.where(raw_pred_left > 0.5)] = 1
            print('1st left')
            raw_pred_left = CCL_check_1ststg(raw_pred_left)

        else:
            raw_ct_left_shape = (raw_ct_shape[0], raw_ct_shape[1], int(raw_ct_shape[2] * 3 / 5))

            raw_pred_left_shape = [raw_ct_shape[0], 200, 200, num_labels_1ststg]
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
                img_ct_left_rs = resample_img_asdim(img_ct_left, net_input_dim, c_val=-1024)
                raw_ct_left_rs = sitk.GetArrayFromImage(img_ct_left_rs)
                raw_ct_left_rs_normed = normalize_vol(raw_ct_left_rs, norm_wind_lower=norm_wind_lower,
                                                      norm_wind_upper=norm_wind_upper)

                fn_raw_ct_left_rs_normed = os.path.join(dir_output_final, 'raw_left_ct_%dx%dx%d.raw' % tuple(
                    reversed(np.shape(raw_ct_left_rs_normed))))
                np.array(raw_ct_left_rs_normed).astype(np.float32).tofile(fn_raw_ct_left_rs_normed)

                raw_ct_left_rs_normed = np.expand_dims(raw_ct_left_rs_normed, axis=0)
                raw_ct_left_rs_normed = np.expand_dims(raw_ct_left_rs_normed, axis=-1)
                prediction = np.squeeze(model_1st.predict(x=raw_ct_left_rs_normed), axis=0)

                raw_pred_left_tmp[z_start:z_start + 200, :, :, :] += prediction
                raw_pred_left_tmp_cnt[z_start:z_start + 200, :, :, :] += 1

            raw_pred_left_tmp[np.where(raw_pred_left_tmp_cnt > 0)] /= raw_pred_left_tmp_cnt[
                np.where(raw_pred_left_tmp_cnt > 0)]
            fn_pred = os.path.join(dir_output_final, 'pred_left.raw')
            np.transpose(raw_pred_left_tmp, [3, 0, 1, 2]).astype(np.float32).tofile(fn_pred)
            if num_labels_1ststg != 1:
                prediction = np.argmax(raw_pred_left_tmp, axis=-1)
            else:
                prediction = np.squeeze(raw_pred_left_tmp)
                prediction[np.where(prediction > 0.5)] = 1

            raw_pred_left = sitk.GetArrayFromImage(
                resample_img_asdim(sitk.GetImageFromArray(prediction), tuple(reversed(raw_ct_left_shape)),
                                   interp=sitk.sitkNearestNeighbor))
            raw_pred_left[np.where(raw_pred_left > 0.5)] = 1
            print('1st left')
            # raw_pred_left = CCL(raw_pred_left, num_labels=2)
            raw_pred_left = CCL_check_1ststg(raw_pred_left)

        # check if both kidneys are valid
        raw_pred_whole = np.zeros(np.shape(raw_ct), dtype=np.uint8)
        raw_pred_right_shape = np.shape(raw_pred_right)
        raw_pred_whole[:, :, :raw_pred_right_shape[2]] = raw_pred_right
        raw_pred_left_shape = np.shape(raw_pred_left)
        raw_pred_left[:, :, :] = raw_pred_left[:, :, -1::-1]
        raw_pred_whole_left_tmp = raw_pred_whole[:, :, -raw_pred_left_shape[2]:]
        raw_pred_whole_left_tmp[np.where(raw_pred_left > 0)] = raw_pred_left[np.where(raw_pred_left > 0)]
        raw_pred_whole[:, :, -raw_pred_left_shape[2]:] = raw_pred_whole_left_tmp
        fn_raw_pred_whole = os.path.join(dir_output_final,
                                         'raw_whole_pred_1st_beforeCCL_%dx%dx%d.raw' % tuple(
                                             reversed(np.shape(raw_pred_whole))))
        raw_pred_whole.astype(np.uint8).tofile(fn_raw_pred_whole)
        raw_pred_whole = CCL_1ststg_post(raw_pred_whole)

        idcs_label_1 = np.where(raw_pred_whole == 1)
        label_1_x_pos = np.mean(idcs_label_1[2])

        idcs_label_2 = np.where(raw_pred_whole == 2)
        print(' # of voxels 1 : %d' % len(idcs_label_1[0]))
        print(' # of voxels 2 : %d' % len(idcs_label_2[0]))
        if len(idcs_label_2[0]) > len(idcs_label_1[0]) * 0.2:
            is_both_kidney = True
            label_2_x_pos = np.mean(idcs_label_2[2])
            print('both kidney')
        else:
            is_both_kidney = False
            print('one kidney')

        if is_both_kidney:
            if label_1_x_pos > label_2_x_pos:
                # swap label btw. 1 and 2
                raw_pred_whole[idcs_label_1] = 2
                raw_pred_whole[idcs_label_2] = 1
                is_left_kidney = True
                is_right_kidney = True
                print('swap position')
            else:
                is_left_kidney = True
                is_right_kidney = True
        else:
            if np.min(idcs_label_1[2]) < raw_ct_shape[2] / 2:
                raw_pred_whole[idcs_label_1] = 1
                raw_pred_whole[idcs_label_2] = 0
                is_right_kidney = True
                is_left_kidney = False
                print('right kidney only')
            else:
                raw_pred_whole[idcs_label_1] = 2
                raw_pred_whole[idcs_label_2] = 0
                is_right_kidney = False
                is_left_kidney = True
                print('left kidney only')

        fn_raw_pred_whole = os.path.join(dir_output_final,
                                         'raw_whole_pred_1st_%dx%dx%d.raw' % tuple(reversed(np.shape(raw_pred_whole))))
        raw_pred_whole.astype(np.uint8).tofile(fn_raw_pred_whole)
        fn_raw_ct_whole = os.path.join(dir_output_final,
                                       'raw_whole_ct_%dx%dx%d.raw' % tuple(reversed(np.shape(raw_pred_whole))))
        raw_ct.astype(np.int16).tofile(fn_raw_ct_whole)

        # extract kidney coordinate
        if is_right_kidney:
            idcs_label_1 = np.where(raw_pred_whole == 1)
            kidney_right_start = (np.max((np.min(idcs_label_1[0] - 16), 0)),
                                  np.max((np.min(idcs_label_1[1] - 16), 0)),
                                  np.max((np.min(idcs_label_1[2] - 16), 0)))
            kidney_right_end = (np.min((np.max(idcs_label_1[0] + 16), raw_ct_shape[0])),
                                np.min((np.max(idcs_label_1[1] + 16), raw_ct_shape[1])),
                                np.min((np.max(idcs_label_1[2] + 16), raw_ct_shape[2])))
            print('kidney_right_start: ' + str(kidney_right_start))
            print('kidney_right_end: ' + str(kidney_right_end))

        if is_left_kidney:
            idcs_label_2 = np.where(raw_pred_whole == 2)
            kidney_left_start = (np.max((np.min(idcs_label_2[0] - 16), 0)),
                                 np.max((np.min(idcs_label_2[1] - 16), 0)),
                                 np.max((np.min(idcs_label_2[2] - 16), 0)))
            kidney_left_end = (np.min((np.max(idcs_label_2[0] + 16), raw_ct_shape[0])),
                               np.min((np.max(idcs_label_2[1] + 16), raw_ct_shape[1])),
                               np.min((np.max(idcs_label_2[2] + 16), raw_ct_shape[2])))

            print('kidney_left_start: ' + str(kidney_left_start))
            print('kidney_left_end: ' + str(kidney_left_end))

        ''' 2nd stage '''

        # Seg right kidney if it is valid
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
            img_ct_right_rs = resample_img_asdim(img_ct_right, net_input_dim, c_val=-1024)
            raw_ct_right_rs = sitk.GetArrayFromImage(img_ct_right_rs)
            raw_ct_right_rs_normed = normalize_vol(raw_ct_right_rs, norm_wind_lower=norm_wind_lower,
                                                   norm_wind_upper=norm_wind_upper)

            fn_raw_ct_right_rs_normed = os.path.join(dir_output_final, 'raw_right_ct_2nd_%dx%dx%d.raw' % tuple(
                reversed(np.shape(raw_ct_right_rs_normed))))
            np.array(raw_ct_right_rs_normed).astype(np.float32).tofile(fn_raw_ct_right_rs_normed)

            raw_ct_right_rs_normed = np.expand_dims(raw_ct_right_rs_normed, axis=0)
            raw_ct_right_rs_normed = np.expand_dims(raw_ct_right_rs_normed, axis=-1)
            prediction = model_2nd.predict(x=raw_ct_right_rs_normed)

            # fn_pred = os.path.join(dir_output_final, 'pred_right.raw')
            # np.transpose(prediction, [0, 4, 1, 2, 3]).astype(np.float32).tofile(fn_pred)

            np.array(raw_ct_right_rs_normed).astype(np.float32).tofile(fn_raw_ct_right_rs_normed)
            if np.shape(prediction)[-1] == 1:
                prediction = np.squeeze(prediction)
            else:
                prediction = np.squeeze(np.argmax(prediction, axis=-1))

            raw_pred_right = sitk.GetArrayFromImage(
                resample_img_asdim(sitk.GetImageFromArray(prediction), tuple(reversed(raw_ct_right_2nd_shape)),
                                   interp=sitk.sitkNearestNeighbor))

            raw_pred_right_tmp = np.array(raw_pred_right)
            raw_pred_right_tmp[np.where(raw_pred_right_tmp > 0)] = 1
            fn_raw_pred_right_tmp = os.path.join(dir_output_final, 'raw_right_pred_beforeCCL_2nd_%dx%dx%d.raw' % tuple(
                reversed(np.shape(raw_pred_right))))
            raw_pred_right_tmp.astype(np.uint8).tofile(fn_raw_pred_right_tmp)
            raw_pred_right_tmp = CCL(raw_pred_right_tmp, num_labels=2)
            raw_pred_right[np.where(raw_pred_right_tmp == 0)] = 0
            fn_raw_pred_right = os.path.join(dir_output_final, 'raw_right_pred_2nd_%dx%dx%d.raw' % tuple(
                reversed(np.shape(raw_pred_right))))
            raw_pred_right.astype(np.uint8).tofile(fn_raw_pred_right)
            fn_raw_ct_right = os.path.join(dir_output_final,
                                           'raw_right_ct_2nd_%dx%dx%d.raw' % tuple(reversed(np.shape(raw_pred_right))))
            raw_ct_right = np.array(raw_ct[kidney_right_start[0]:kidney_right_end[0],
                                    kidney_right_start[1]:kidney_right_end[1],
                                    kidney_right_start[2]:kidney_right_end[2]])
            raw_ct_right.astype(np.int16).tofile(fn_raw_ct_right)

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
            img_ct_left_rs = resample_img_asdim(img_ct_left, net_input_dim, c_val=-1024)
            raw_ct_left_rs = sitk.GetArrayFromImage(img_ct_left_rs)
            raw_ct_left_rs_normed = normalize_vol(raw_ct_left_rs, norm_wind_lower=norm_wind_lower,
                                                  norm_wind_upper=norm_wind_upper)

            fn_raw_ct_left_rs_normed = os.path.join(dir_output_final, 'raw_left_ct_2nd_%dx%dx%d.raw' % tuple(
                reversed(np.shape(raw_ct_left_rs_normed))))
            np.array(raw_ct_left_rs_normed).astype(np.float32).tofile(fn_raw_ct_left_rs_normed)

            raw_ct_left_rs_normed = np.expand_dims(raw_ct_left_rs_normed, axis=0)
            raw_ct_left_rs_normed = np.expand_dims(raw_ct_left_rs_normed, axis=-1)
            prediction = model_2nd.predict(x=raw_ct_left_rs_normed)

            # fn_pred = os.path.join(dir_output_final, 'pred_left.raw')
            # np.transpose(prediction, [0, 4, 1, 2, 3]).astype(np.float32).tofile(fn_pred)

            np.array(raw_ct_left_rs_normed).astype(np.float32).tofile(fn_raw_ct_left_rs_normed)
            if np.shape(prediction)[-1] == 1:
                prediction = np.squeeze(prediction)
            else:
                prediction = np.squeeze(np.argmax(prediction, axis=-1))

            raw_pred_left = sitk.GetArrayFromImage(
                resample_img_asdim(sitk.GetImageFromArray(prediction), tuple(reversed(raw_ct_left_2nd_shape)),
                                   interp=sitk.sitkNearestNeighbor))
            raw_pred_left = raw_pred_left[:, :, -1::-1]

            raw_pred_left_tmp = np.array(raw_pred_left)
            raw_pred_left_tmp[np.where(raw_pred_left_tmp > 0)] = 1
            fn_raw_pred_left_tmp = os.path.join(dir_output_final, 'raw_left_pred_beforeCCL_2nd_%dx%dx%d.raw' % tuple(
                reversed(np.shape(raw_pred_left))))
            raw_pred_left_tmp.astype(np.uint8).tofile(fn_raw_pred_left_tmp)
            raw_pred_left_tmp = CCL(raw_pred_left_tmp, num_labels=2)
            raw_pred_left[np.where(raw_pred_left_tmp == 0)] = 0
            fn_raw_pred_left = os.path.join(dir_output_final, 'raw_left_pred_2nd_%dx%dx%d.raw' % tuple(
                reversed(np.shape(raw_pred_left))))
            raw_pred_left.astype(np.uint8).tofile(fn_raw_pred_left)
            fn_raw_ct_left = os.path.join(dir_output_final,
                                          'raw_left_ct_2nd_%dx%dx%d.raw' % tuple(reversed(np.shape(raw_pred_left))))
            raw_ct_left = np.array(raw_ct[kidney_left_start[0]:kidney_left_end[0],
                                   kidney_left_start[1]:kidney_left_end[1],
                                   kidney_left_start[2]:kidney_left_end[2]])
            raw_ct_left.astype(np.int16).tofile(fn_raw_ct_left)

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

        fn_raw_pred_whole = os.path.join(dir_output_final,
                                         'raw_whole_pred_2nd_%dx%dx%d.raw' % tuple(reversed(np.shape(raw_pred_whole))))
        raw_pred_whole.astype(np.uint8).tofile(fn_raw_pred_whole)

        fn_raw_pred_whole_color = os.path.join(dir_output_final, 'color_raw_whole_pred_2nd_%dx%dx%d.raw' % tuple(
            reversed(np.shape(raw_pred_whole))))
        # colorize_boundary_and_fill_vol(raw_ct, raw_pred_whole, norm_wind_lower=-200, norm_wind_upper=300,
        #                                fn_out_fullpath=fn_raw_pred_whole_color)

        if pid == 223:
            raw_pred_whole_tmp = np.zeros(np.shape(raw_ct_original), dtype=np.uint8)
            raw_pred_whole_tmp[-180:, :, :] = raw_pred_whole
            raw_pred_whole = raw_pred_whole_tmp
        ''' Save final prediction as Nifti '''
        fn_final_pred_nifti = os.path.join(dir_output_base, dir_output, 'prediction_%05d.nii.gz' % pid)
        x_nib = nib.load(fn_img_ct)
        p_nib = nib.Nifti1Image(raw_pred_whole[-1::-1], x_nib.affine)
        nib.save(p_nib, fn_final_pred_nifti)

        # sitk.WriteImage(img_final_pred, fn_final_pred_nifti, True)
        if raw_gt is not None:
            fn_csv = os.path.join(dir_output_final, 'Dice.csv')
            GetDice(raw_gt, raw_pred_whole, fn_csv, num_label=3)

            fn_raw_gt = os.path.join(dir_output_final, 'raw_whole_gt_%dx%dx%d.raw' % tuple(reversed(np.shape(raw_gt))))
            raw_gt.astype(np.uint8).tofile(fn_raw_gt)

            fn_raw_gt_whole_color = os.path.join(dir_output_final,
                                                 'color_raw_whole_gt_%dx%dx%d.raw' % tuple(reversed(np.shape(raw_gt))))
            # colorize_boundary_and_fill_vol(raw_ct, raw_gt, norm_wind_lower=-200, norm_wind_upper=300,
            #                                fn_out_fullpath=fn_raw_gt_whole_color)
    summary_dice(os.path.join(dir_output_base, dir_output))


def TransAxis(img, dtype):
    spacing = img.GetSpacing()
    direction = img.GetDirection()

    img_raw = sitk.GetArrayFromImage(img)  # order x y z
    img_raw = np.transpose(img_raw, axes=[2, 1, 0])
    img_raw = img_raw[-1::-1, :, :]
    img_new = sitk.GetImageFromArray(img_raw.astype(dtype))
    img_new.SetSpacing(tuple(reversed(spacing)))
    img_new.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return img_new


def TransAxisReverse(raw_data, ref_img, dtype):
    spacing = ref_img.GetSpacing()
    direction = ref_img.GetDirection()
    origin = ref_img.GetOrigin()

    raw_data_flip = raw_data[-1::-1, :, :]
    raw_data_tr = np.transpose(raw_data_flip, axes=[2, 1, 0])

    print('raw_data_tr shape: ' + str(np.shape(raw_data_tr)))
    # img_new = sitk.GetImageFromArray(raw_data_tr.astype(dtype=dtype))
    # img_new.SetSpacing(spacing)
    # img_new.SetDirection(direction)
    # img_new.SetOrigin(origin)

    # return img_new

    return raw_data_tr


def resample_img_asdim(img, target_size_itkorder, interp=sitk.sitkLinear, c_val=0):
    img_size = img.GetSize()
    img_spacing = img.GetSpacing()

    target_spacing = ((img_size[0]) / target_size_itkorder[0] * img_spacing[0],
                      (img_size[1]) / target_size_itkorder[1] * img_spacing[1],
                      (img_size[2]) / target_size_itkorder[2] * img_spacing[2])

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetInterpolator(interp)
    resampler.SetSize(target_size_itkorder)
    resampler.SetDefaultPixelValue(c_val)
    return resampler.Execute(img)


def normalize_vol(vol, norm_wind_lower, norm_wind_upper):
    slope = norm_wind_upper - norm_wind_lower
    vol = vol - norm_wind_lower
    vol = vol / slope
    vol[np.where(vol < 0)] = 0
    vol[np.where(vol > 1)] = 1

    return vol


def CCL(vol, num_labels):
    structure_label = np.ones(shape=(3, 3, 3))
    new_vol = np.zeros(shape=np.shape(vol)).astype(np.uint8)
    for idx in range(1, num_labels):
        # print('CCL_processing [%d]' % idx)
        vol_binary = np.zeros(np.shape(vol)).astype(np.uint16)
        vol_binary[np.where(vol == idx)] = 1

        labeled_vol, max_num_compo = scipy.ndimage.label(vol_binary, structure=structure_label)
        # print('max_num_compo: %d' % max_num_compo)
        if max_num_compo > 70:
            # print('continue')
            continue

        volume_each_compo = [0]
        for idx_comp in range(1, max_num_compo + 1):
            volume_each_compo.append(len(np.where(labeled_vol == idx_comp)[0]))

        idx_for_maxvol = np.argmax(volume_each_compo)
        new_vol[np.where(labeled_vol == idx_for_maxvol)] = idx

    return new_vol


def CCL_check_1ststg(vol):
    structure_label = np.ones(shape=(3, 3, 3))
    new_vol = np.array(vol)

    new_vol[np.where(new_vol > 0.5)] = 1
    labeled_vol, max_num_compo = scipy.ndimage.label(new_vol, structure=structure_label)
    if max_num_compo > 70:
        new_vol[:, :, :] = 0

    # for idx in range(1, num_labels):
    #     print('CCL_processing [%d]' % idx)
    #     vol_binary = np.zeros(np.shape(vol)).astype(np.uint16)
    #     vol_binary[np.where(vol == idx)] = 1
    #
    #     labeled_vol, max_num_compo = scipy.ndimage.label(vol_binary, structure=structure_label)
    #     print('max_num_compo: %d' % max_num_compo)
    #     if max_num_compo > 50:
    #         print('continue')
    #         continue
    #
    #     else:
    #         new_vol[np.where(labeled_vol == idx_for_maxvol)] = idx

        # volume_each_compo = [0]
        # for idx_comp in range(1, max_num_compo + 1):
        #     volume_each_compo.append(len(np.where(labeled_vol == idx_comp)[0]))
        #
        # idx_for_maxvol = np.argmax(volume_each_compo)
        # new_vol[np.where(labeled_vol == idx_for_maxvol)] = idx

    return new_vol


def CCL_1ststg_post(vol):
    structure_label = np.ones(shape=(3, 3, 3))
    new_vol = np.zeros(shape=np.shape(vol)).astype(np.uint8)
    idx = 1
    vol_binary = np.zeros(np.shape(vol)).astype(np.uint16)
    vol_binary[np.where(vol == idx)] = 1

    labeled_vol, max_num_compo = scipy.ndimage.label(vol_binary, structure=structure_label)
    volume_each_compo = [0]
    for idx_comp in range(1, max_num_compo + 1):
        volume_each_compo.append(len(np.where(labeled_vol == idx_comp)[0]))

    idx_for_maxvol = np.argmax(volume_each_compo)
    new_vol[np.where(labeled_vol == idx_for_maxvol)] = idx
    volume_each_compo[idx_for_maxvol] = 0
    idcs_first = np.where(labeled_vol == idx_for_maxvol)
    # one more time

    if max_num_compo > 1:
        idx_for_maxvol = np.argmax(volume_each_compo)
        idcs_second = np.where(labeled_vol == idx_for_maxvol)
        if len(idcs_second[0]) > len(idcs_first[0]) * 0.2:
            # print('idcs_first volume: %d' % len(idcs_first[0]))
            # print('idcs_second volume: %d' % len(idcs_second[0]))
            new_vol[np.where(labeled_vol == idx_for_maxvol)] = idx + 1

    return new_vol


def summary_dice(DIR_SRC='\\\\data5\\Workspace\\yongjin\\KidneySeg\\eval\\ACE_C_v1_seed_1000-ckpt-240-ACE_C_2ndstg_v1_seed_1108-ckpt-490'):
    pid_list = list(range(0, 5)) + list(range(6, 15)) + list(range(16, 20)) + [40, 165, 194]

    dice_pid = []
    dice_kidney = []
    dice_cancer = []
    for pid in pid_list:
        fn_dice = os.path.join(DIR_SRC, 'case_%05d' % pid, 'Dice.csv')
        dice_tmp = []
        with open(fn_dice, 'r') as fid:
            rdr = csv.reader(fid)
            for line in rdr:
                dice_tmp.append(float(line[1]))

        dice_kidney.append(dice_tmp[0])
        dice_cancer.append(dice_tmp[1])
        dice_pid.append('case_%03d' % pid)

    print(dice_kidney)
    print(dice_cancer)

    fn_summary_csv = os.path.join(DIR_SRC, 'Dice_summary.csv')
    with open(fn_summary_csv, 'w', newline='') as fid:
        writer = csv.writer(fid)
        writer.writerow(dice_pid)
        writer.writerow(dice_kidney)
        writer.writerow(dice_cancer)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % args.dev
    # os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % 0
    main()
    # summary_dice()
