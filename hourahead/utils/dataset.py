import os
import pandas as pd
import numpy as np
import torch

from torch.utils.data.dataset import Dataset

class Dataset(object):
    
    def __init__(self, path, timesteps):
        self.data_path = path
        self.timesteps = timesteps
        self.train_data_raw = pd.read_excel(os.path.join(path, "hour_ahead/train_in.xlsx")).values
        self.train_target_raw = pd.read_excel(os.path.join(path, "hour_ahead/train_out.xlsx")).values
        self.test_data_raw = pd.read_excel(os.path.join(path, "hour_ahead/test_in.xlsx")).values
        self.test_target_raw = pd.read_excel(os.path.join(path, "hour_ahead/test_out.xlsx")).values
        self.train_data, self.train_target = self.process_train()
        self.test_data, self.test_target = self.process_test()
        min_max = pd.read_excel(os.path.join(self.data_path, "hour_ahead/max_min.xls"))
        self._min = float(min_max["pmin"][0])
        self._max = float(round(min_max["pmax"][0], 2))
    
    def process_train(self):
        train_data, temp_row = list(), list()
        for i in range(len(self.train_data_raw) - (self.timesteps - 1)):
            for j in range(self.timesteps):
                temp_row.append(list(self.train_data_raw[i + j]))
            
            train_data.append(temp_row)
            temp_row = list()
        
        train_target = self.train_target_raw[self.timesteps-1:]

        return train_data, train_target

    def process_test(self):
        test_data, temp_row = list(), list()
        for i in range(len(self.test_data_raw) - (self.timesteps - 1)):
            for j in range(self.timesteps):
                temp_row.append(list(self.test_data_raw[i + j]))
            
            test_data.append(temp_row)
            temp_row = list()
        
        test_target = self.test_target_raw[self.timesteps-1:]

        return test_data, test_target

    
class energyDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
        
    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float(), torch.from_numpy(self.target[index]).float()
        
    def __len__(self):
        return len(self.data)