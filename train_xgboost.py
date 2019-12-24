import pandas as pd
import xgboost as xgb
import numpy as np
import math, os
from sklearn import datasets, linear_model, metrics
from sklearn.metrics import mean_squared_error, r2_score, make_scorer


base_path = os.path.dirname(os.path.abspath(__file__))


train_data = pd.read_excel(os.path.join(base_path, "data/hour_ahead/train_in.xlsx"))
train_target = pd.read_excel(os.path.join(base_path, "data/hour_ahead/train_out.xlsx"))
test_data = pd.read_excel(os.path.join(base_path, "data/hour_ahead/test_in.xlsx"))
test_target = pd.read_excel(os.path.join(base_path, "data/hour_ahead/test_out.xlsx"))

train_target_max = 5.583
train_target_min = 0

trainset_list = train_df.values[:, ::2]
testset_list  = test_df.values[:, ::2]

train_target_ori = (train_target * (train_target_max - train_target_min)) + train_target_min  # 訓練資料label反正規
test_target_ori = (test_target * (train_target_max - train_target_min)) + train_target_min  # 測試資料label反正規
# print(train_target)
# print('test_target.values: ', test_target.values)


#模型建構


regr = xgb.XGBRegressor( 
    gamma=0.5,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=3,
    n_estimators=1976,
    reg_alpha=0.9,
    reg_lambda=0.7,
    eval_metric='rmse',
    objective='reg:logistic',
    subsample=0.2,
    seed=88,
)

# print('XGBRegressor')
# XGBoost training

preds = regr.predict(train_data)  # predict results
preds = (preds * (train_target_max - train_target_min)) + train_target_min  # 預測結果反正規
# print('predic:', preds)
target = train_target_ori.values
# print('train_target_ori:', train_target_ori)
# print('target:', target)
#儲存
# regr.save_model('test.model')

print(pd.DataFrame({'predict': preds}).shape)
print(pd.DataFrame({'predict': preds}))

#評估模型
print('R2_score = ')
print(metrics.r2_score(target, preds))
print('\nMSE = ')
print(metrics.mean_squared_error(target, preds))
print('\nRMSE = ')
print(math.sqrt(metrics.mean_squared_error(target, preds)))
print('\nMAE = ')
print(metrics.mean_absolute_error(target, preds))
 

