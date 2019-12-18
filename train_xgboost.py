import pandas as pd
import xgboost as xgb
import numpy as np
import math, os
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer


base_path = os.path.dirname(os.path.abspath(__file__))


train_data = pd.read_excel(os.path.join(base_path, "data/hour_ahead/train_in.xlsx"))
train_target = pd.read_excel(os.path.join(base_path, "data/hour_ahead/train_out.xlsx"))
test_data = pd.read_excel(os.path.join(base_path, "data/hour_ahead/test_in.xlsx"))
test_target = pd.read_excel(os.path.join(base_path, "data/hour_ahead/test_out.xlsx"))

train_target_max = 5.583
train_target_min = 0

train_target_ori = (train_target * (train_target_max - train_target_min)) + train_target_min  # 訓練資料label反正規
test_target_ori = (test_target * (train_target_max - train_target_min)) + train_target_min  # 測試資料label反正規
# print(train_target)
# print('test_target.values: ', test_target.values)


#模型建構


def denorm_rmse(target, pred):
    pred = (pred * (train_target_max - train_target_min) + train_target_min)
    return math.sqrt(metrics.mean_squared_error(target, pred))

parameters = {
    'max_depth': list(range(1, 11)),
    'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
    'n_estimators': [500, 1000, 2000, 3000, 5000],
    # 'min_child_weight': list(range(1, 21)),
    # 'subsample': list(np.arange(0.1, 1.1, 0.1)),
    # 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
    # 'reg_alpha': list(np.arange(0.1, 1.1, 0.1)),
    # 'reg_lambda': list(np.arange(0.1, 1.1, 0.1)),
    # 'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1],
    "seed": list(range(10, 160, 10)),
    # "gamma": list(np.arange(0.1, 1.1, 0.1))
}

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

gsearch = GridSearchCV(regr, param_grid=parameters, scoring=make_scorer(denorm_rmse), cv=3)
gsearch.fit(train_data, train_target)
print("Best score: %0.3f" % gsearch.best_score_)
 

