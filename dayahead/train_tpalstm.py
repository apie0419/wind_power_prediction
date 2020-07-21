from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from utils import Dataset, energyDataset
from tpalstm import TPALSTM

import torch, torch.nn.functional as F
import os, pandas as pd, numpy as np

GPU           = 6
batch_size    = 64
hidden_units  = 24
learning_rate = 0.002
n_layers      = 5
epoches       = 300
output_dims   = 8
timesteps     = 8
num_input     = 22

torch.cuda.set_device(GPU)

base_path = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(base_path, "../data/1")

dataset = Dataset(data_path, timesteps)

train_data, train_target = np.array(dataset.train_data).astype(np.float32), np.array(dataset.train_target).astype(np.float32)
trainloader = DataLoader(energyDataset(train_data, train_target), batch_size=batch_size, shuffle=True, num_workers=0)

model = TPALSTM(num_input, output_dims, hidden_units, timesteps * 2 + 3, n_layers)
model = model.cuda()

optimizer = Adam(model.parameters(), lr=learning_rate)

pbar = tqdm(range(1, epoches+1))

model.train()

losses = list()

for epoch in pbar:
    if epoch % 100 == 0:
        learning_rate *= 0.7
        optimizer = Adam(model.parameters(), lr=learning_rate)

    train_loss = 0.

    for data, target in trainloader:
        data, target = data.cuda(), target.cuda()
        preds = model(data)
        loss = F.mse_loss(preds, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    
    train_loss /= len(trainloader.dataset)
    train_loss = round(train_loss * 100, 2)

    # train_preds, train_target = list(), list()
    # for i in range(0, len(dataset.train_data), 8):
    #     logits = None
    #     if i + 10 > len(dataset.train_data):
    #         break
    #     for j in range(11):
    #         x, y = dataset.train_data[i + j], dataset.train_target[i + j]
    #         if type(logits) != type(None):
    #             x[-1][-1] = float(logits.data.cpu().numpy()[0][0])
    #         x = torch.from_numpy(np.array(x).astype(np.float32)).float().cuda()
    #         x = x.view((1, timesteps, num_input))
            
    #         logits = model(x)
    #         if j > 2:
    #             train_preds.append(logits.data.cpu().numpy()[0][0])
    #             train_target.append(y)

    # train_preds = np.array(train_preds)
    # train_target = np.array(train_target)
    # train_loss = round(np.sqrt(np.mean(np.square(train_target - train_preds))) * 100., 2)

    losses.append(train_loss)
    pbar.set_description(f"Epoch {epoch}, Loss: {train_loss}")

model.eval()

with torch.no_grad():

    # test_preds, test_target = list(), list()
    # for i in range(0, len(dataset.test_data), 8):
    #     logits = None
    #     if i + 10 > len(dataset.test_data):
    #         break
    #     for j in range(11):
    #         x, y = dataset.test_data[i + j], dataset.test_target[i + j]
    #         if type(logits) != type(None):
    #             x[-1][-1] = float(logits.data.cpu().numpy()[0][0])
    #         x = torch.from_numpy(np.array(x).astype(np.float32)).float().cuda()
    #         x = x.view((1, timesteps, num_input))
            
    #         logits = model(x)
    #         if j > 2:
    #             test_preds.append(logits.data.cpu().numpy()[0][0])
    #             test_target.append(y)

    # test_preds = np.array(test_preds)
    # test_target = np.array(test_target)
    # test_loss = round(np.sqrt(np.mean(np.square(test_target - test_preds))) * 100., 2)
    train_data = torch.from_numpy(np.array(dataset.train_data).astype(np.float32)).float()
    train_target = np.array(dataset.train_target).astype(np.float32)
    train_data = train_data.cuda()
    train_preds = model(train_data)
    train_preds = train_preds.data.cpu().numpy()
    # print (train_preds.shape, train_target.shape)
    train_loss = round(np.sqrt(np.mean(np.square(train_target - train_preds))) * 100., 2)
    test_data = torch.from_numpy(np.array(dataset.test_data).astype(np.float32)).float()
    test_target = np.array(dataset.test_target).astype(np.float32)
    test_data = test_data.cuda()
    test_preds = model(test_data)
    test_preds = test_preds.data.cpu().numpy()
    test_loss = round(np.sqrt(np.mean(np.square(test_target - test_preds))) * 100., 2)

print (f"Train RMSE Loss: {train_loss}%, Test RMSE Loss: {test_loss}%")

if not os.path.exists(os.path.join(base_path, "Output")):
    os.mkdir(os.path.join(base_path, "Output"))

pd.DataFrame({
    "loss": losses
}).plot()

plt.savefig(os.path.join(base_path, "Output/tpalstm_loss.png"))

pd.DataFrame({
    "predict": test_preds.flatten(),
    "target": test_target.flatten()
}).plot()

plt.savefig(os.path.join(base_path, "Output/tpalstm_evaluation.png"))