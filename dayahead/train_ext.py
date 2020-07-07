import pandas as pd
import numpy as np
import os, math

from matplotlib              import pyplot as plt
from utils                   import Dataset, denorm, rmse
from sklearn                 import datasets, linear_model, metrics
from sklearn.ensemble        import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV

base_path = os.path.dirname(os.path.abspath(__file__))

timesteps = 8
num_input = 23

data_path = os.path.join(base_path, "../data/1")
dataset = Dataset(data_path, timesteps)
_min, _max = dataset._min, dataset._max

train_data = np.reshape(dataset.train_data, (-1, num_input * timesteps))
test_data = np.reshape(dataset.test_data, (-1, num_input * timesteps))

model = ExtraTreesRegressor(
    n_estimators=2000,
    criterion= "mse",
    max_features=20,
    max_depth=15,
    warm_start=True,
    max_samples=0.8,
    ccp_alpha=0,
    n_jobs=-1
)
print("Training...")

model.fit(train_data, dataset.train_target)

print("Finish")

train_target = denorm(dataset.train_target, _min, _max)
test_target = denorm(dataset.test_target, _min, _max)

predict, target = list(), list()
for i in range(0, len(train_data), 8):
    logits = None
    if i + 10 > len(train_data):
        break
    for j in range(11):
        x, y = train_data[i + j], train_target[i + j]
        if logits != None:
            x[-1] = float(logits[0])
        
        x = np.reshape(x, (1, timesteps * num_input))
        logits = model.predict(x)
        if j > 2:
            denorm_x = denorm(logits, _min, _max)
            predict.append(denorm_x[0])
            target.append(y)

predict = np.array(predict, dtype=np.float32)
target = np.array(target, dtype=np.float32)
train_loss = math.sqrt(metrics.mean_squared_error(target, predict))
print("Train RMSE: {:.2f}%".format(train_loss / _max * 100.))

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
        logits = model.predict(x)
        if j > 2:
            denorm_x = denorm(logits, _min, _max)
            predict.append(denorm_x[0])
            target.append(y)

predict = np.array(predict, dtype=np.float32)
target = np.array(target, dtype=np.float32)
test_loss = math.sqrt(metrics.mean_squared_error(target, predict))
print("Test RMSE: {:.2f}%".format(test_loss / _max * 100.))

pd.DataFrame({
    "predict": predict,
    "target": target
}).plot()

plt.savefig(os.path.join(base_path, "Output/extratree_evaluation.png"))