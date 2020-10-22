import os
import sys
import random
import numpy as np
import SimpleITK as sitk
from scipy import ndimage


class Preprocessing:
    mean_std = [-52.453253428141274, 37.56369903352525]
    windowing_range = [-40., 120.]

    def __init__(self,
                 args,
                 rotation_range=[10., 10., 10.],
                 voi_dict=None):

        self.task = args.task
        self.standard = args.standard
        self.classes = args.classes
        self.rotation_range = rotation_range
        self.voi_dict = voi_dict

        self.windowing_min = self.windowing_range[0] - self.windowing_range[1] // 2
        self.windowing_max = self.windowing_range[0] + self.windowing_range[1] // 2

    def _getvoi(self, xx):
        img_dummy, mask_dummy, data = xx

        if self.task == 'FINE1':
            voisize = [100, 80, 80]
            center = [(self.voi_dict[data][1] + self.voi_dict[data][0]) // 2,
                      (self.voi_dict[data][3] + self.voi_dict[data][2]) // 2,
                      (self.voi_dict[data][5] + self.voi_dict[data][4]) // 2]
            print('\n', self.voi_dict[data][1] - self.voi_dict[data][0],
                  self.voi_dict[data][3] - self.voi_dict[data][2], self.voi_dict[data][5] - self.voi_dict[data][4])
            print('voisize :', voisize, 'center :', center, 'img :', img_dummy.shape)

            img = []
            for i in range(8):
                img.append(img_dummy[center[0] - voisize[0] * (1 - i // 4):center[0] + voisize[0] * (i // 4),
                           center[1] - voisize[1] * (1 - i % 2):center[1] + voisize[1] * (i % 2),
                           center[2] - voisize[2] * (1 - (i // 2) % 2):center[2] + voisize[2] * ((i // 2) % 2)])


            # random.shuffle(img)
            # random.shuffle(mask)

        elif self.task == 'FINE2':
            voisize = [200, 180, 180]
            center = [(self.voi_dict[data][1] + self.voi_dict[data][0]) // 2,
                      (self.voi_dict[data][3] + self.voi_dict[data][2]) // 2,
                      (self.voi_dict[data][5] + self.voi_dict[data][4]) // 2]
            img = img_dummy[center[0] - voisize[0] // 2:center[0] + voisize[0] // 2,
                  center[1] - voisize[1] // 2:center[1] + voisize[1] // 2,
                  center[2] - voisize[2] // 2:center[2] + voisize[2] // 2]

        elif self.task == 'FINE3':
            pass

        elif self.task == 'FINE':
            img = img_dummy[self.voi_dict[data][0]:self.voi_dict[data][1] + 1,
                  self.voi_dict[data][2]:self.voi_dict[data][3] + 1, self.voi_dict[data][4]:self.voi_dict[data][5] + 1]

        else:
            raise ValueError

        return img

    def _dilation(self, mask):
        dilate_filter = sitk.BinaryDilateImageFilter()
        dilate_filter.SetKernelRadius([1, 1, 1])
        dilate_filter.SetForegroundValue(1)
        mask = dilate_filter.Execute(mask)
        return mask

    def _resize(self, xx):
        img = xx
        img = ndimage.zoom(img, [.5, .5, .5], order=1, mode='constant', cval=0.)
        return img

    def _array2img(self, xx):
        img = xx
        img = sitk.GetArrayFromImage(img).astype('float32')
        return img

    def _windowing(self, x):
        return np.clip(x, self.windowing_min, self.windowing_max)

    def _expand(self, xx):
        img = xx
        img = img[np.newaxis, ..., np.newaxis]
        return img

    # def _merge_mask(self, x):
    #     x[x != 0] = 1.
    #     return x

    def _onehot(self, x):
        result = np.zeros(x.shape + (self.classes,))
        for i in range(self.classes):
            result[..., i][np.where(x == i)] = 1.
        return result

    def _horizontal_flip(self, xx, axis=2):
        img = xx
        if np.random.random() < 0.5:
            def flip(x):
                x = np.asarray(x).swapaxes(axis, 0)
                x = x[::-1, ...]
                x = x.swapaxes(0, axis)
                return x

            img = flip(img)

        return img

    def _standardize(self, x):
        if self.standard == "norm":
            return (x - self.mean_std[0]) / self.mean_std[1]
        elif self.standard == "eachnorm":
            return (x - x.mean()) / x.std()
        elif self.standard == "minmax":
            return (x - self.windowing_min) / (self.windowing_max - self.windowing_min)
        else:
            return x

    def _rotation(self, xx, dep_index=0, row_index=1, col_index=2, fill_mode='nearest', cval=0.):
        img = xx

        theta1 = np.pi / 180 * np.random.uniform(-self.rotation_range[0], self.rotation_range[0])
        theta2 = np.pi / 180 * np.random.uniform(-self.rotation_range[1], self.rotation_range[1])
        theta3 = np.pi / 180 * np.random.uniform(-self.rotation_range[2], self.rotation_range[2])

        rotation_matrix_z = np.array([[np.cos(theta1), -np.sin(theta1), 0, 0],
                                      [np.sin(theta1), np.cos(theta1), 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
        rotation_matrix_y = np.array([[np.cos(theta2), 0, -np.sin(theta2), 0],
                                      [0, 1, 0, 0],
                                      [np.sin(theta2), 0, np.cos(theta2), 0],
                                      [0, 0, 0, 1]])
        rotation_matrix_x = np.array([[1, 0, 0, 0],
                                      [0, np.cos(theta3), -np.sin(theta3), 0],
                                      [0, np.sin(theta3), np.cos(theta3), 0],
                                      [0, 0, 0, 1]])

        rotation_matrix = np.dot(np.dot(rotation_matrix_y, rotation_matrix_z), rotation_matrix_x)

        d, h, w = img.shape[dep_index], img.shape[row_index], img.shape[col_index]
        transform_matrix = self.__transform_matrix_offset_center(rotation_matrix, d, w, h)
        img = self.__apply_transform(img, transform_matrix, fill_mode, cval)

        return img

    def __transform_matrix_offset_center(self, matrix, x, y, z):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        o_z = float(z) / 2 + 0.5
        offset_matrix = np.array([[1, 0, 0, o_x],
                                  [0, 1, 0, o_y],
                                  [0, 0, 1, o_z],
                                  [0, 0, 0, 1]])
        reset_matrix = np.array([[1, 0, 0, -o_x],
                                 [0, 1, 0, -o_y],
                                 [0, 0, 1, -o_z],
                                 [0, 0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix

    def __apply_transform(self, x, transform_matrix, fill_mode='nearest', cval=0.):
        final_affine_matrix = transform_matrix[:3, :3]
        final_offset = transform_matrix[:3, 3]
        x = ndimage.interpolation.affine_transform(x,
                                                   final_affine_matrix,
                                                   final_offset,
                                                   order=0,
                                                   mode=fill_mode,
                                                   cval=cval)
        return x



