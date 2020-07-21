import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data.dataset import Dataset

class Dataset(object):
    
    def __init__(self, path, timesteps):
        self.data_path = path
        self.timesteps = timesteps
        self.train_data_raw = pd.read_excel(os.path.join(path, "day_ahead/train_in.xlsx"), header=None).values
        self.train_target_raw = pd.read_excel(os.path.join(path, "day_ahead/train_out.xlsx"), header=None).values
        self.test_data_raw = pd.read_excel(os.path.join(path, "day_ahead/test_in.xlsx"), header=None).values
        self.test_target_raw = pd.read_excel(os.path.join(path, "day_ahead/test_out.xlsx"), header=None).values
        self.train_data, self.train_target = self.process_train()
        self.test_data, self.test_target = self.process_test()
        min_max = pd.read_excel(os.path.join(self.data_path, "day_ahead/max_min.xls"))
        self._min = float(min_max["pmin"][0])
        self._max = float(round(min_max["pmax"][0], 2))
    
    def process_train(self):
        
        train_data, train_target = list(), list()

        for i in range(self.timesteps + 3, len(self.train_data_raw) - (self.timesteps - 1), 8):
            temp_data_row = list()
            temp_target_row = list()
            for j in range(self.timesteps):
                temp_data_row.append(list(self.train_data_raw[i + j][4:]))
                temp_target_row.append(self.train_target_raw[i + j][4])
            for j in range(1, self.timesteps + 4):
                temp_data_row.append(list(self.train_data_raw[i - j][4:]))
            train_data.append(temp_data_row)
            train_target.append(temp_target_row)

        # train_target = self.train_target_raw[self.timesteps - 1:, 4]
        return train_data, train_target
        

    def process_test(self):
        
        test_data, test_target = list(), list()

        for i in range(self.timesteps + 3, len(self.test_data_raw) - (self.timesteps - 1), 8):
            temp_data_row = list()
            temp_target_row = list()
            for j in range(self.timesteps):
                temp_data_row.append(list(self.test_data_raw[i + j][4:]))
                temp_target_row.append(self.test_target_raw[i + j][4])
            for j in range(1, self.timesteps + 4):
                temp_data_row.append(list(self.test_data_raw[i - j][4:]))

            test_data.append(temp_data_row)
            test_target.append(temp_target_row)

        # test_data, temp_data_row = list(), list()
        # for i in range(8, len(self.test_data_raw) - (self.timesteps - 1), 8):
        #     for j in range(self.timesteps):
        #         temp_data_row.append(list(self.test_data_raw[i + j][4:]))
        #     test_data.append(temp_data_row)
        #     temp_data_row = list()
            
        # test_target = self.test_target_raw[self.timesteps - 1:, 4]

        return test_data, test_target


class energyDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
        
    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float(), torch.from_numpy(self.target[index]).float()
        
    def __len__(self):
        return len(self.data)