import os
import pandas as pd
import numpy as np

class Dataset(object):
    
    def __init__(self, path, timesteps, output_dim):
        self.data_path = path
        self.timesteps = timesteps
        self.output_dim = output_dim
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
        
        train_data, train_target, temp_data_row, temp_target_row = list(), list(), list(), list()
        for i in range(len(self.train_data_raw) - (self.timesteps + self.output_dim - 1)):
            for j in range(self.timesteps):
                temp_data_row.append([self.train_data_raw[i + j][4], self.train_data_raw[i + j][-1], self.train_target_raw[i + j][4]])
            
            for j in range(i + self.timesteps + 3, i + self.timesteps + self.output_dim):
                temp_target_row.append(self.train_target_raw[j][4])

            train_data.append(temp_data_row)
            train_target.append(temp_target_row)
            temp_data_row = list()
            temp_target_row = list()

        return train_data, train_target

    def process_test(self):
        test_data, test_target, temp_data_row, temp_target_row = list(), list(), list(), list()
        for i in range(5, len(self.test_data_raw) - (self.output_dim + self.timesteps - 1), self.timesteps):
            print ("Data: ")

            for j in range(self.timesteps):
                print (self.train_data_raw[i + j][:4])
                temp_data_row.append([self.test_data_raw[i + j][4], self.test_data_raw[i + j][-1], self.test_target_raw[i + j][4]])
            
            for j in range(i + self.timesteps + 3, i + self.timesteps + self.output_dim):
                temp_target_row.append(self.train_target_raw[j][4])

            test_data.append(temp_data_row)
            test_target.append(temp_target_row)
            temp_data_row = list()
            temp_target_row = list()
            
        return test_data, test_target
        # test_data, test_target, temp_data_row, temp_target_row = list(), list(), list(), list()
        # for i in range(len(self.test_data_raw) - (self.output_dim + self.timesteps - 1)):
        #     for j in range(self.timesteps):
        #         temp_data_row.append([self.test_data_raw[i + j][4], self.test_data_raw[i + j][-1], self.test_target_raw[i + j][4]])
        #         temp_target_row.append(self.test_target_raw[i + j + self.timesteps][4])
        #     test_data.append(temp_data_row)
        #     test_target.append(temp_target_row)
        #     temp_data_row = list()
        #     temp_target_row = list()
        
        # return test_data, test_target

    def append(self, dataset):
        pass

class OneByOneDataset(object):
    
    def __init__(self, path):
        self.data_path = path
        self.timesteps = timesteps
        self.train_data_raw = pd.read_excel(os.path.join(path, "day_ahead/train_in.xlsx")).values
        self.train_target_raw = pd.read_excel(os.path.join(path, "day_ahead/train_out.xlsx")).values
        self.test_data_raw = pd.read_excel(os.path.join(path, "day_ahead/test_in.xlsx")).values
        self.test_target_raw = pd.read_excel(os.path.join(path, "day_ahead/test_out.xlsx")).values
        self.train_data, self.train_target = self.process_train()
        self.test_data, self.test_target = self.process_test()
        min_max = pd.read_excel(os.path.join(self.data_path, "day_ahead/max_min.xls"))
        self._min = float(min_max["pmin"][0])
        self._max = float(round(min_max["pmax"][0], 2))
    
    def process_train(self):
        train_data, train_target = list(), list()

        for i in range(len(self.train_data_raw) - 1):
            train_data.append(list(self.train_data_raw[i][4:]) + [self.train_target_raw[i][4]])
            train_target.append(self.train_target_raw[i+1][4])
            
        return train_data, train_target

    def process_test(self):
        test_data, test_target, temp_data_row, temp_target_row = list(), list(), list(), list()
        for i in range(len(self.test_data_raw) - self.timesteps):
            for j in range(self.timesteps):
                temp_data_row.append(list(self.test_data_raw[i + j][4:]) + [self.test_target_raw[i + j][4]])
                temp_target_row.append(self.test_target_raw[i + j + self.timesteps][4])

            test_data.append(temp_data_row)
            test_target.append(temp_target_row)
            temp_data_row = list()
            temp_target_row = list()
        
        return test_data, test_target

    def append(self, dataset):
        pass
    