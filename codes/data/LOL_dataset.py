from torch.utils.data import Dataset
from glob import glob
import random
import os
import data.util as util
import numpy as np
import torchvision.transforms as TF
def get_len(route, phase):
    if phase =='train':
        train_low_data_names = glob(route + 'low/*.png')
        train_low_data_names.sort()
        train_high_data_names = glob(route + 'high/*.png')
        train_high_data_names.sort()
        return len(train_high_data_names),train_low_data_names,train_high_data_names
    elif phase =='test':
        test_low_data_names = glob(route + 'low/*.png')
        test_low_data_names.sort()
        test_high_data_names = glob(route + 'high/*.png')
        test_high_data_names.sort()
        return len(test_low_data_names), test_low_data_names,test_high_data_names
    elif phase == 'eval':
        eval_low_data_names = glob(route + 'low/*.png')
        eval_low_data_names.sort()
        eval_high_data_names = glob(route + 'high/*.png')
        eval_high_data_names.sort()
        return len(eval_high_data_names), eval_low_data_names,eval_high_data_names
    else:
        return 0, []

class LOLDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.route = opt['route']
        self.phase = opt['phase']
        self.patch_size = opt['patch_size']
        self.input_images = [None]*500
        self.gt_images = [None]*500
        self.num = [0]*250
        self.pre = [0]*250
        self.len, self.low_names,self.high_names = get_len(self.route, self.phase)
        print(len(self.low_names))
        if self.phase == 'train':
            for ids in range(self.len):
                self.gt_images[ids] = util.read_img(self.high_names[ids])
                self.input_images[ids] = util.read_img(self.low_names[ids])

    def __getitem__(self,index):
        if self.phase == 'train':
            train_low_data = self.input_images[index]
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
            high_im = util.read_img(self.high_names[index])
            # BGR to RGB, HWC to CHW, numpy to tensor
            if high_im.shape[2] == 3:
                low_im = low_im[:, :, [2, 1, 0]]
                high_im = high_im[:, :, [2, 1, 0]]
            return {'LQ': TF.ToTensor()(low_im), 'GT': TF.ToTensor()(high_im), 'raw_path': self.low_names[index]}
        elif self.phase == 'eval':
            low_im = util.read_img(self.low_names[index])
            high_im = util.read_img(self.high_names[index])
            if high_im.shape[2] == 3:
                low_im = low_im[:, :, [2, 1, 0]]
                high_im = high_im[:, :, [2, 1, 0]]
            return {'LQ': TF.ToTensor()(low_im), 'GT': TF.ToTensor()(high_im), 'raw_path': self.low_names[index]}
    def __len__(self):
        return self.len
