import pandas as pd
import xgboost as xgb
import numpy as np
import os, math
from matplotlib import pyplot as plt
from utils import Dataset, denorm, rmse
from sklearn import datasets, linear_model, metrics
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

base_path = os.path.dirname(os.path.abspath(__file__))

timesteps = 8
num_input = 6

data_path = os.path.join(base_path, "../data/1")
dataset = Dataset(data_path, timesteps)
_min, _max = dataset._min, dataset._max

train_data = np.reshape(dataset.train_data, (-1, num_input * timesteps))
test_data = np.reshape(dataset.test_data, (-1, num_input * timesteps))

#模型建構

regr = xgb.XGBRegressor(
    gamma=0,
    learning_rate=0.01,
    max_depth=20,
    min_child_weight=10,
    n_estimators=2000,
    reg_alpha=0.5,
    reg_lambda=0.7,
    eval_metric='rmse',
    objective='reg:logistic',
    subsample=0.8,
    seed=50
)

regr.fit(train_data, dataset.train_target)

train_target = denorm(dataset.train_target, _min, _max)
test_target = denorm(dataset.test_target, _min, _max)

predict, target = list(), list()
for i in range(0, len(test_data), 8):
    logits = None
    if i + 10 > len(test_data):
        break
    for j in range(11):
        x, y = test_data[i + j], test_target[i + j]
        if logits != None:
            x[-1] = float(logits[0])
        x = np.reshape(x, (1, timesteps * num_input))
        logits = regr.predict(x)
        if j > 2:
            denorm_x = denorm(logits, _min, _max)
            predict.append(denorm_x[0])
            target.append(y)

predict = np.array(predict, dtype=np.float32)
target = np.array(target, dtype=np.float32)
print (predict.shape)
print (target.shape)
test_loss = rmse(predict, target)
print(math.sqrt(metrics.mean_squared_error(target, predict)))

# preds = regr.predict(test_data) 
# preds = denorm(preds, _min, _max)
# target = test_target[:, 0]

# #評估模型
# print('\nTest RMSE = ')
# print(math.sqrt(metrics.mean_squared_error(target, preds)))

pd.DataFrame({
    "predict": predict,
    "target": target
}).plot()

plt.savefig(os.path.join(base_path, "Output/xgboost_evaluation.png"))