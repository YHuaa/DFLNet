import os
import warnings
import math
import h5py
import scipy.io as scio
import numpy as np
from torch.utils.data import Dataset
from utils.utils import data_preprocess
# ignore warnings
warnings.filterwarnings("ignore")


class SoundDataset(Dataset):
    def __init__(self, data_dir, label_dir):
        super(SoundDataset, self).__init__()
        self.data_name = []
        self.label_dir = label_dir
        self.position_x_y = []
        with open(data_dir, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.data_name.append(line)

        with open('./data/sound_source_data_fixed_distance/NewDataset/One/One.txt', 'r') as f:
        # with open(data_dir.replace(data_dir.split('/')[-1], data_dir.split('/')[-2] + '.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.position_x_y.append(line)

        
    def __len__(self):
        return len(self.data_name)

    def __getitem__(self, index):
        f = h5py.File(self.data_name[index], 'r')
        raw_sound_data = np.array(f['time_data'][()])
        
        position_x_y_index = int(self.data_name[index].split('/')[-1].split('.')[0].split('_')[-1]) - 1
        label_position = []
        label_position.append(self.position_x_y[position_x_y_index].split('_')[2])
        label_position.append(self.position_x_y[position_x_y_index].split('_')[3])
        label_position = [float(x) for x in label_position]
        # label_position_x = self.position_x_y[position_x_y_index].split('_')[2]
        # label_position_y = self.position_x_y[position_x_y_index].split('_')[3]


        label = scio.loadmat(self.label_dir + 'GT_' + self.data_name[index].split('/')[-2] + '/' 
        + self.data_name[index].split('/')[-1].split('.')[-2] + '.mat')['B']
        label = label.reshape(label.size, 1, order='F') # order = 'F' 按照列排序

        _, ATA, ATb, DAS_results, label_coordinate_to_vector_pos = data_preprocess(raw_sound_data, label_position)

        label_coordinate_to_vector_pos = np.array(label_coordinate_to_vector_pos)
        label_coordinate_to_vector_pos = label_coordinate_to_vector_pos.reshape(1, label_coordinate_to_vector_pos.size)

        # print("########SAMPLE", self.data_name[index])
        # print("########1", label[label_coordinate_to_vector_pos[0][0]])
        # print("########2", label_coordinate_to_vector_pos[0][0])
        # print("########3", np.max(label))
        return ATA, ATb, DAS_results, label, label_coordinate_to_vector_pos, self.data_name[index].split('/')[-1].split('.')[-2]


if __name__ == '__main__':
    data_dir = './data/One_train.txt'
    label_dir = './data/sound_source_data_fixed_distance/NewLabel/GT_One/'
    s = SoundDataset(data_dir, label_dir)
    s.__getitem__(1)