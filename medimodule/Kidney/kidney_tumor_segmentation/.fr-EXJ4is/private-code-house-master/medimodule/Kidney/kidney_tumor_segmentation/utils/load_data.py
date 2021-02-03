import os
# import json
import random
# import pandas as pd
import numpy as np
import threading
import SimpleITK as sitk
from scipy import ndimage

from sklearn.model_selection import train_test_split

INPUT_FOLDER = '../../../data/interpolated'


def load(seed=42, test_size=0.1):
    if not os.path.isfile('./trainset.txt'):
        cases = sorted([c for c in os.listdir(INPUT_FOLDER) if not 'case_00005' in c and os.path.isdir(os.path.join(INPUT_FOLDER, c))])
        idx_train, idx_test = train_test_split(range(1, len(cases)+1), test_size=test_size, random_state=seed)
        
        trainset = [cases[x-1] for x in idx_train]
        testset = [cases[x-1] for x in idx_test]

        # with open('./trainset.txt', 'w') as f:
        #     for t in trainset:
        #         f.write(t+'\n')
        # with open('./testset.txt', 'w') as f:
        #     for v in testset:
        #         f.write(v+'\n')

    else:
        with open('./trainset.txt', 'r') as f:
            trainset = f.readlines()
            trainset = [t[:-1] for t in trainset]
        with open('./testset.txt', 'r') as f:
            testset = f.readlines()
            testset = [v[:-1] for v in testset]

    print('# of training data :', len(trainset), ', # of test data :', len(testset))
    return trainset, testset


##############################################
# Generator
##############################################
class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def Generator(datalist,
              task,
              mode='train',
              standard='minmax',
              rotation_range=[10., 10., 10.],
              wlevel=100.,
              wwidth=400.,
              seed=42,
              shuffle=True):

    random.seed(seed)
    def _preprocessing(img, mask, prep, idx=None):
        img, mask = prep._array2img([img, mask])
        img, mask, flag = prep._getvoi([img, mask, idx])
        
        if flag:
            if idx == 1 and task == 'tumor':
                img, mask = prep._horizontal_flip([img, mask])
            elif np.random.rand(1) > .5 and task != 'tumor':
                img, mask = prep._horizontal_flip([img, mask])
            
            if mode == 'train':
                img, mask = prep._rotation([img, mask])
            img = prep._windowing(img)
            img = prep._standard(img)
            mask = prep._onehot(mask)
            img, mask = prep._expand([img, mask])
        
        return [img, mask], flag
            

    prep = Preprocessing(task=task,
                         standard=standard,
                         wlevel=wlevel,
                         wwidth=wwidth,
                         rotation_range=rotation_range)

    while True:
        if shuffle:
            random.shuffle(datalist)
        
        for data in datalist:
            # data = 'case_00151'
            if mode == 'test':
                print(data)
            img = sitk.ReadImage(os.path.join(INPUT_FOLDER, data, 'imaging.nii'))
            mask = sitk.ReadImage(os.path.join(INPUT_FOLDER, data, 'segmentation.nii'))
            if task == 'tumor':
                for idx in range(2):
                    result, flag = _preprocessing(img, mask, prep, idx)
                    if flag:
                        yield result

            else:
                 result, flag = _preprocessing(img, mask, prep, 0)
                 yield result


class Preprocessing:
    def __init__(self,
                 task,
                 standard='minmax',
                 wlevel=50.,
                 wwidth=300.,
                 rotation_range=[10., 10., 10.]):
        
        self.task = task
        self.standard = standard
        self.rotation_range = rotation_range

        self.classes = 3
        self.windowing_min = wlevel - wwidth//2
        self.windowing_max = wlevel + wwidth//2

    def _array2img(self, xx, istest=False):
        img, mask = xx
        if istest:
            spacing = img.GetSpacing()
        img = sitk.GetArrayFromImage(img).astype('float32')
        mask = sitk.GetArrayFromImage(mask).astype('float32')
        if istest:
            return img, mask, spacing
        else:
            return img, mask

    def _expand(self, xx):
        img, mask = xx
        img = img[np.newaxis,...,np.newaxis]
        mask = mask[np.newaxis,...]
        return img, mask

    def _standard(self, x):
        if self.standard == 'minmax':
            x = (x - x.min()) / (x.max() - x.min())
        elif self.standard == 'normal':
            x = (x - 80.96544691699005) / 58.84357050328374
        else:
            pass
        return x

    def _windowing(self, x):
        x = np.clip(x, self.windowing_min, self.windowing_max)
        return x

    def _onehot(self, x):
        result = np.zeros(x.shape+(self.classes,))
        for i in range(self.classes):
            result[...,i][np.where(x == i)] = 1.
        return result

    def _horizontal_flip(self, xx):
        img, mask = xx
        return img[::-1], mask[::-1]

    def _getvoi(self, xx, istest=False):
        img, mask, idx = xx

        if self.task == 'tumor':
            cut = mask.copy()
            if idx == 0:
                cut[int(cut.shape[0]//2):,:] = 0
            else:
                cut[:-int(cut.shape[0]//2),:] = 0
            
            if cut.sum() == 0:
                flag = False
                if istest:
                    return img, cut, flag, []
                else:
                    return img, cut, flag
            else:
                flag = True

            bbox = [0, cut.shape[0], 0, cut.shape[1], 0, cut.shape[2]]
            for a in range(cut.shape[0]):
                if cut[a,:,:].sum() != 0:
                    bbox[0] = a
                    break

            for a in range(cut.shape[0]-1, 0, -1):
                if cut[a,:,:].sum() != 0:
                    bbox[1] = a
                    break

            for c in range(cut.shape[1]):
                if cut[:,c,:].sum() != 0:
                    bbox[2] = c
                    break

            for c in range(cut.shape[1]-1, 0, -1):
                if cut[:,c,:].sum() != 0:
                    bbox[3] = c
                    break

            for s in range(cut.shape[2]):
                if cut[:,:,s].sum() != 0:
                    bbox[4] = s
                    break

            for s in range(cut.shape[2]-1, 0, -1):
                if cut[:,:,s].sum() != 0:
                    bbox[5] = s
                    break

            img = img[np.maximum(0, bbox[0]):np.minimum(img.shape[0]-1, bbox[1]+1),
                      np.maximum(0, bbox[2]):np.minimum(img.shape[1]-1, bbox[3]+1),
                      np.maximum(0, bbox[4]):np.minimum(img.shape[2]-1, bbox[5]+1)]
            mask = mask[np.maximum(0, bbox[0]):np.minimum(mask.shape[0]-1, bbox[1]+1),
                        np.maximum(0, bbox[2]):np.minimum(mask.shape[1]-1, bbox[3]+1),
                        np.maximum(0, bbox[4]):np.minimum(mask.shape[2]-1, bbox[5]+1)]

            if istest:
                return img, mask, flag, bbox
            else:
                return img, mask, flag

        elif self.task == 'tumor1':
            threshold = [380, 230, 72]
            bbox = [0, mask.shape[0], 0, mask.shape[1], 0, mask.shape[2]]
            for a in range(mask.shape[0]):
                if mask[a,:,:].sum() != 0:
                    bbox[0] = a
                    break

            for a in range(mask.shape[0]-1, 0, -1):
                if mask[a,:,:].sum() != 0:
                    bbox[1] = a
                    break

            for c in range(mask.shape[1]):
                if mask[:,c,:].sum() != 0:
                    bbox[2] = c
                    break

            for c in range(mask.shape[1]-1, 0, -1):
                if mask[:,c,:].sum() != 0:
                    bbox[3] = c
                    break

            for s in range(mask.shape[2]):
                if mask[:,:,s].sum() != 0:
                    bbox[4] = s
                    break

            for s in range(mask.shape[2]-1, 0, -1):
                if mask[:,:,s].sum() != 0:
                    bbox[5] = s
                    break

            flag = True
            center = [int(bbox[0]+bbox[1])//2,int(bbox[2]+bbox[3])//2,int(bbox[4]+bbox[5])//2]
            bbox = [center[0]-int(threshold[0]//2), center[0]+int(threshold[0]//2),
                    center[1]-int(threshold[1]//2), center[1]+int(threshold[1]//2),
                    center[2]-int(threshold[2]//2), center[2]+int(threshold[2]//2)]

            img = img[np.maximum(0, bbox[0]):np.minimum(img.shape[0], bbox[1]),
                      np.maximum(0, bbox[2]):np.minimum(img.shape[1], bbox[3]),
                      np.maximum(0, bbox[4]):np.minimum(img.shape[2], bbox[5])]
            mask = mask[np.maximum(0, bbox[0]):np.minimum(mask.shape[0], bbox[1]),
                        np.maximum(0, bbox[2]):np.minimum(mask.shape[1], bbox[3]),
                        np.maximum(0, bbox[4]):np.minimum(mask.shape[2], bbox[5])]

            diff = [(bbox[1]-bbox[0])-img.shape[0],
                    (bbox[3]-bbox[2])-img.shape[1],
                    (bbox[5]-bbox[4])-img.shape[2]]

            diff = [d if d % 2 == 0 else d+1 for d in diff]
            # print(diff, img.shape, mask.shape, bbox, bbox[5]-bbox[4])
            img = np.pad(img, ((diff[0]//2,diff[0]//2),(diff[1]//2,diff[1]//2),(diff[2]//2,diff[2]//2)), 'minimum')
            mask = np.pad(mask, ((diff[0]//2,diff[0]//2),(diff[1]//2,diff[1]//2),(diff[2]//2,diff[2]//2)), 'minimum')

            diff1 = [0, 0, 0] # shrink
            if img.shape[0] > bbox[1]-bbox[0]:
                img = img[:-1]
                mask = mask[:-1]
                diff1[0] -= 1
            if img.shape[1] > bbox[3]-bbox[2]:
                img = img[:,:-1]
                mask = mask[:,:-1]
                diff1[1] -= 1
            if img.shape[2] > bbox[5]-bbox[4]:
                img = img[:,:,:-1]
                mask = mask[:,:,:-1]
                diff1[2] -= 1

            if istest:
                return img, mask, flag, bbox, diff, diff1
            else:
                return img, mask, flag

    def _rotation(self, xx, theta=None, dep_index=0, row_index=1, col_index=2, fill_mode='nearest', cval=0.):
        img, mask = xx

        if theta:
            theta1, theta2, theta3 = theta
        else:
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
        mask = self.__apply_transform(mask, transform_matrix, fill_mode, cval)

        return img, mask

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


if __name__ == "__main__":
    trainset, testset = load()
    valset = trainset[:len(testset)]
    trainset = trainset[len(testset):]
    print('  --> # of training data :', len(trainset), '/ # of validation data :', len(valset))