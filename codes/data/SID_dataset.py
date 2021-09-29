from torch.utils.data import Dataset
from glob import glob
import random
import torch
import os
import data.util as util
import numpy as np
import rawpy
import torchvision.transforms as TF

def get_len(route, phase):
    if phase =='train':
        train_low_data_names = glob(route + 'Sony/train_low/0*_00*.png')
        train_low_data_names.sort()
        train_low_data_names1 = glob(route + 'Fuji/train_low/0*_00*.png')
        train_low_data_names1.sort()
        train_low_data_names.extend(train_low_data_names1)
        train_high_data_names = glob(route + 'Sony/train_high/0*_00*.png')
        train_high_data_names.sort()
        train_high_data_names1 = glob(route + 'Fuji/train_high/0*_00*.png')
        train_high_data_names1.sort()
        train_high_data_names.extend(train_high_data_names1)
        return len(train_high_data_names),train_low_data_names,train_high_data_names
    elif phase =='test':
        test_low_data_names = glob(route + 'Sony/test_low/1*_00*.png')
        test_low_data_names.sort()
        # test_low_data_names1 = glob(route + 'Fuji/test_low/1*_00*.png')
        # test_low_data_names1.sort()
        # test_low_data_names.extend(test_low_data_names1)
        test_high_data_names = glob(route + 'Sony/test_high/1*_00*.png')
        test_high_data_names.sort()
        # test_high_data_names1 = glob(route + 'Fuji/test_high/1*_00*.png')
        # test_high_data_names1.sort()
        # test_high_data_names.extend(test_high_data_names1)
        return len(test_low_data_names), test_low_data_names,test_high_data_names
    elif phase == 'eval':
        eval_low_data_names = glob(route + 'Sony/eval_low/1000*_00*.png')
        eval_low_data_names.sort()
        eval_low_data_names1 = glob(route + 'Sony/eval_low/1001*_00*.png')
        eval_low_data_names1.sort()
        eval_low_data_names.extend(eval_low_data_names1)
        eval_high_data_names = glob(route+'Sony/eval_high/1000*_00*.png')
        eval_high_data_names.sort()
        eval_high_data_names1 = glob(route + 'Sony/eval_high/1001*_00*.png')
        eval_high_data_names1.sort()
        eval_low_data_names.extend(eval_low_data_names1)
        return len(eval_low_data_names), eval_low_data_names,eval_high_data_names
    else:
        return 0, []

def pack_raw(raw):
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

class SIDDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.route = opt['route']
        self.phase = opt['phase']
        self.patch_size = opt['patch_size']
        self.input_images = [None]*500
        self.gt_images = [None]*250
        self.num = [0]*250
        self.pre = [0]*250
        self.len, self.low_names,self.high_names = get_len(self.route, self.phase)
        if self.phase == 'train':
            j = 0
            for ids in range(self.len):
                self.gt_images[ids] = util.read_img(self.high_names[ids])
                idstr = os.path.basename(self.high_names[ids])[0:5]
                gt_exposure = float(os.path.basename(self.high_names[ids])[9:-5])
                self.pre[ids] = j
                while j < len(self.low_names):
                    if idstr == os.path.basename(self.low_names[j])[0:5] :
                        in_exposure = float(os.path.basename(self.low_names[j])[9:-5])
                        ratio = min(gt_exposure/in_exposure,300)
                        self.input_images[j] = util.read_img(self.low_names[j])
                        self.num[ids] += 1
                        j += 1
                    else :
                        break

    def __getitem__(self,index):
        if self.phase == 'train':
            randomx = random.randint(0,self.num[index]-1)
            train_low_data = self.input_images[self.pre[index]+randomx]
            train_high_data = self.gt_images[index]
            h,w,_ = train_low_data.shape
            x = random.randint(0,h-self.patch_size)
            y = random.randint(0,w-self.patch_size)
            low_im = train_low_data[x:x+self.patch_size,y:y+self.patch_size,:]
            high_im = train_high_data[x:x+self.patch_size,y:y+self.patch_size,:]
            if np.random.randint(2, size=1)[0] == 1:  # random flip
                low_im = np.flip(low_im, axis=0)
                high_im = np.flip(high_im, axis=0)
            if np.random.randint(2, size=1)[0] == 1:
                low_im = np.flip(low_im, axis=1)
                high_im = np.flip(high_im, axis=1)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                low_im = np.transpose(low_im, (1, 0, 2))
                high_im = np.transpose(high_im, (1, 0, 2))
            # BGR to RGB, HWC to CHW, numpy to tensor
            if high_im.shape[2] == 3:
                low_im = low_im[:, :, [2, 1, 0]]
                high_im = high_im[:, :, [2, 1, 0]]
            return {'LQ': TF.ToTensor()(low_im.copy()), 'GT': TF.ToTensor()(high_im.copy())}
        elif self.phase == 'test':
            low_im = util.read_img(self.low_names[index])
            idstr = os.path.basename(self.low_names[index])
            for ids in range(len(self.high_names)):
                tepstr = os.path.basename(self.high_names[ids])
                if idstr[0:5] == tepstr[0:5]:
                    high_im = util.read_img(self.high_names[ids])
                    # BGR to RGB, HWC to CHW, numpy to tensor
                    if high_im.shape[2] == 3:
                        low_im = low_im[:, :, [2, 1, 0]]
                        high_im = high_im[:, :, [2, 1, 0]]
                    return {'LQ': TF.ToTensor()(low_im), 'GT': TF.ToTensor()(high_im), 'raw_path': self.low_names[index]}
        elif self.phase == 'eval':
            low_im = util.read_img(self.low_names[index])
            idstr = os.path.basename(self.low_names[index])
            for ids in range(len(self.high_names)):
                tepstr = os.path.basename(self.high_names[ids])
                if idstr[0:5] == tepstr[0:5]:
                    high_im = util.read_img(self.high_names[ids])
                    # BGR to RGB, HWC to CHW, numpy to tensor
                    if high_im.shape[2] == 3:
                        low_im = low_im[:, :, [2, 1, 0]]
                        high_im = high_im[:, :, [2, 1, 0]]
                    return {'LQ': TF.ToTensor()(low_im), 'GT': TF.ToTensor()(high_im)}

    def __len__(self):
        return self.len
