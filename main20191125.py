import pandas as pd
import xgboost as xgb
import numpy as np
import math, os
from sklearn import datasets, linear_model, metrics
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

base_path = os.path.dirname(os.path.abspath(__file__))

input_rows = 24

train_data_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/train_in.xlsx"))
train_target_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/train_out.xlsx"))
test_data_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/test_in.xlsx"))
test_target_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/test_out.xlsx"))
train_target_max = 5.583
train_target_min = 0

## 直接每個 row 訓練

# train_data = train_data_df.values
# train_target = train_target_df.values
# test_data = test_data_df.values
# test_target = test_target_df.values

##

train_data_list = train_data_df.values[:, ::2]
test_data_list  = test_data_df.values[:, ::2]


train_data = list()
test_data = list()
train_target = list()
test_target = list()


temp_row = list()

for i in range(len(train_data_list) - (input_rows - 1)):
    for j in range(input_rows):
        temp_row += list(train_data_list[i + j])
    
    train_data.append(temp_row)
    temp_row = list()

temp_row = list()

for i in range(len(test_data_list) - (len(test_data_list) % input_rows)):
    for j in range(input_rows):
        temp_row += list(test_data_list[i + j])
    test_data.append(temp_row)
    temp_row = list()

train_target = train_target_df.values[input_rows-1:]
test_target = test_target_df.values[input_rows-1:]


train_target_ori = (train_target * (train_target_max - train_target_min)) + train_target_min 
test_target_ori = (test_target * (train_target_max - train_target_min)) + train_target_min  


#模型建構

regr = xgb.XGBRegressor(
    gamma=0.1,
    learning_rate=0.01,
    max_depth=10,
    min_child_weight=10,
    n_estimators=2000,
    reg_alpha=0.5,
    reg_lambda=0.7,
    eval_metric='rmse',
    objective='reg:logistic',
    subsample=0.8,
    seed=50
)

# XGBoost training
regr.fit(train_data, train_target)

#預測

preds = regr.predict(train_data) 
preds = (preds * (train_target_max - train_target_min)) + train_target_min 
target = train_target_ori
# target = target[:, 0]

#評估模型
print('\nTrain RMSE = ')
print(math.sqrt(metrics.mean_squared_error(target, preds)))

preds = regr.predict(test_data) 
preds = (preds * (train_target_max - train_target_min)) + train_target_min 
target = test_target_ori
target = target[:, 0]

#評估模型
print('\nTest RMSE = ')
print(math.sqrt(metrics.mean_squared_error(target, preds)))