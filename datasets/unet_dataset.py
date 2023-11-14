import os
import warnings
import math
import h5py
import scipy.io as scio
import numpy as np
from torch.utils.data import Dataset
import sys
import cv2
# sys.path.append('..')
sys.path.append('..')
from utils import data_preprocess, generate_grey_img
# ignore warnings
warnings.filterwarnings("ignore")


class ImageDataset(Dataset):
    def __init__(self, data_dir, DAS_img_save_dir, label_img_save_dir):
        super(ImageDataset, self).__init__()
        self.data_name = []
        self.DAS_img_save_dir = DAS_img_save_dir
        self.label_img_save_dir = label_img_save_dir

        with open(data_dir, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.data_name.append(line)
        
    def __len__(self):
        return len(self.data_name)

    def __getitem__(self, index):
        DAS_results_img = cv2.imread(self.DAS_img_save_dir + self.data_name[index].split('/')[-2] + 
        '/' + self.data_name[index].split('/')[-1].split('.')[0] + '.png')
        label_img = cv2.imread(self.label_img_save_dir + 'GT_' + self.data_name[index].split('/')[-2] + 
        '/' + self.data_name[index].split('/')[-1].split('.')[0] + '.png')

        DAS_results_img = DAS_results_img.transpose(2, 0, 1)
        label_img = label_img.transpose(2, 0, 1)

        DAS_results_img = DAS_results_img[0, :, :]
        label_img = label_img[0, :, :]

        DAS_results_img = DAS_results_img.reshape(1, DAS_results_img.shape[0], DAS_results_img.shape[1])
        label_img = label_img.reshape(1, label_img.shape[0], label_img.shape[1])

        return DAS_results_img, label_img, self.data_name[index].split('/')[-1].split('.')[0]

if __name__ == '__main__':
    data_dir = './data/One_train.txt'
    DAS_img_save_dir = './data/sound_source_data_fixed_distance/DAS_reults/'
    label_img_save_dir = './data/sound_source_data_fixed_distance/label/'
    s = ImageDataset(data_dir, DAS_img_save_dir, label_img_save_dir)
    s.__getitem__(10)