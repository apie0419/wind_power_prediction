import pandas as pd
import matplotlib.pyplot as plt
import os

base_path = os.path.dirname(os.path.abspath(__file__))

train_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/train_in.xlsx"))
train_label_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/train_out.xlsx"))
test_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/test_in.xlsx"))
test_label_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/test_out.xlsx"))

train_target_max = 5.583
train_target_min = 0

train_df = (train_df * (train_target_max - train_target_min)) + train_target_min  # 訓練資料label反正規
# test_target_ori = (test_target * (train_target_max - train_target_min)) + train_target_min

t_1_power = train_df.values[:, 0]
t_wind = train_df.values[:, 2]


df = pd.DataFrame({
    "t-1_power": t_1_power,
})

df2 = pd.DataFrame({
    "t_wind": t_wind,
})


df.plot()
plt.show()
df2.plot()
plt.show()