import os
import numpy as np
import tifffile as tiff
import skimage
import imageio as iio

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import random

def center_crop(imm, size, imtype = 'image'):
    h = int(size[0]/2)
    w = int(size[1]/2)
    ch = int(imm.shape[0]/2)
    cw = int(imm.shape[1]/2)
    if imtype == 'image':
        return imm[ch-h:ch+h, cw-w:cw+w, :]
    else:
        return imm[ch-h:ch+h, cw-w:cw+w]

class Dataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.    
    """
    
    def __init__(
            self, 
            root,
            resize = False,
            size = (384,384),
            scale_factor = 2,
            augmentation= False,
            task = 'regression',
            d_min = 30,
            dataset_name = 'BCDD',
            phase = 'train',
            train_list = ['class12', 'class13', 'class14', 'class15', 'class16','class21','class22','class24','class25'],
            test_list = ['class11', 'class23','class26']
        ):
        self.phase = phase
        self.train_list = train_list
        self.test_list = test_list
        self.t1_images_dir_list = []
        self.t2_images_dir_list = []
        self.masks2d_dir_list = []
        
        self.cla_num = [0]     #记录不同类的idx分界线

        self.t1_images_dir_total_list = []     #所有的图片的完整路径
        self.t2_images_dir_total_list = []
        self.masks2d_dir_total_list = []

        if self.phase=='train':
            for class_i in self.train_list:     
                self.t1_images_dir_list.append(os.path.join(root, class_i, 'time1/'))
                self.t2_images_dir_list.append(os.path.join(root, class_i, 'time2/'))
                self.masks2d_dir_list.append(os.path.join(root, class_i, 'label/'))
        else:
            for test_i in self.test_list:
                self.t1_images_dir_list.append(os.path.join(root, test_i, 'time1/'))
                self.t2_images_dir_list.append(os.path.join(root, test_i, 'time2/'))
                self.masks2d_dir_list.append(os.path.join(root, test_i, 'label/'))
        
        if self.phase=='train':
            for i in range(len(self.train_list)):
                self.t1_images_ids = os.listdir(self.t1_images_dir_list[i])
                self.t2_images_ids = os.listdir(self.t2_images_dir_list[i])
                self.masks2d_ids = os.listdir(self.masks2d_dir_list[i])

                self.t1_images_fps = [os.path.join(self.t1_images_dir_list[i], image_id) for image_id in self.t1_images_ids]
                self.t2_images_fps = [os.path.join(self.t2_images_dir_list[i], image_id) for image_id in self.t2_images_ids]
                self.masks2d_fps = [os.path.join(self.masks2d_dir_list[i], image_id) for image_id in self.masks2d_ids]

                self.t1_images_dir_total_list.extend(self.t1_images_fps)
                self.t2_images_dir_total_list.extend(self.t2_images_fps)
                self.masks2d_dir_total_list.extend(self.masks2d_fps)

                self.cla_num.append(self.cla_num[i] + len(self.t1_images_fps)) 

        else:
            for i in range(len(self.test_list)):
                self.t1_images_ids = os.listdir(self.t1_images_dir_list[i])
                self.t2_images_ids = os.listdir(self.t2_images_dir_list[i])
                self.masks2d_ids = os.listdir(self.masks2d_dir_list[i])

                self.t1_images_fps = [os.path.join(self.t1_images_dir_list[i], image_id) for image_id in self.t1_images_ids]
                self.t2_images_fps = [os.path.join(self.t2_images_dir_list[i], image_id) for image_id in self.t2_images_ids]
                self.masks2d_fps = [os.path.join(self.masks2d_dir_list[i], image_id) for image_id in self.masks2d_ids]
                
                self.t1_images_dir_total_list.extend(self.t1_images_fps)
                self.t2_images_dir_total_list.extend(self.t2_images_fps)
                self.masks2d_dir_total_list.extend(self.masks2d_fps)

                self.cla_num.append(self.cla_num[i] + len(self.t1_images_fps))

        self.ids = self.t1_images_dir_total_list
        
        self.augmentation = augmentation
        self.resize = resize
        self.size = size
        self.scale_factor = scale_factor
        self.task = task
        self.d_min = d_min
        self.dataset_name = dataset_name

    def __getitem__(self, i):
        
        # read data with tifffile because of 3d mask int16
        qt1 = iio.imread(self.t1_images_dir_total_list[i])#.transpose([2,0,1])
        qt2 = iio.imread(self.t2_images_dir_total_list[i])#.transpose([2,0,1])
        qmask = iio.imread(self.masks2d_dir_total_list[i])
        qmask = qmask[:,:,0]
        qmask = np.where(qmask == 255,1,0)
        

        for j in range((len(self.cla_num)-1)):
            if i>=self.cla_num[j] and i<self.cla_num[j+1]:
                while True:
                    support_id = random.randint(self.cla_num[j], self.cla_num[j+1]-1)
                    if support_id != i:
                        break
                break

        st1 = iio.imread(self.t1_images_dir_total_list[support_id])#.transpose([2,0,1])
        st2 = iio.imread(self.t2_images_dir_total_list[support_id])#.transpose([2,0,1])
        smask = iio.imread(self.masks2d_dir_total_list[support_id])
        smask = smask[:,:,0]
        smask = np.where(smask == 255,1,0)
        
        # if self.dataset_name == 'LEVIR':
        #     mask2d = (mask2d == 255)+0
        # elif self.dataset_name == 'BCDD':
        #     shape = mask2d.shape
        #     a = torch.ones(tuple(shape))
        #     mask2d = np.where(mask2d>0,a,mask2d)  
            
        # if self.resize == 'resize': #Parametro resize?
        #     t1 = skimage.transform.resize(t1, self.size, order = 0, preserve_range=True)
        #     t2 = skimage.transform.resize(t2, self.size, order = 0, preserve_range=True)
        #     mask2d = skimage.transform.resize(mask2d, self.size, order = 0, preserve_range=True)
            
        # elif self.resize == 'crop':
        #     t1 = center_crop(t1, size = self.size)
        #     t2 = center_crop(t2, size = self.size)
        #     mask2d = center_crop(mask2d, size = self.size, imtype = 'mask')
            
        # apply augmentations
        if self.augmentation:
            qsample = self.augmentation(image=qt1, t2=qt2, mask=qmask)
            qt1, qt2, qmask = qsample['image'], qsample['t2'], qsample['mask']
            
            ssample = self.augmentation(image=st1, t2=st2, mask=smask)
            st1, st2, smask = ssample['image'], ssample['t2'], ssample['mask']


        return st1, st2, qt1, qt2, smask, qmask

        
    def __len__(self):
        return len(self.ids)