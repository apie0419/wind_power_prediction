import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

base_path = os.path.dirname(os.path.abspath(__file__))

train_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/train_in.xlsx"))
train_label_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/train_out.xlsx"))
test_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/test_in.xlsx"))
test_label_df = pd.read_excel(os.path.join(base_path, "data/hour_ahead/test_out.xlsx"))

train_target_max = 5583
train_target_min = 0

# train_df = (train_df * (train_target_max - train_target_min)) + train_target_min  
# test_df = (test_df * (train_target_max - train_target_min)) + train_target_min
# test_target_ori = (test_target * (train_target_max - train_target_min)) + train_target_min


# twenty_four = train_df.values[:-24, 0]
# twelve = train_df.values[12:-12, 0]
# six = train_df.values[18:-6, 0]
# three = train_df.values[21:-3, 0]
# two = train_df.values[22:-2, 0]
# one = train_df.values[23:-1, 0]
# now = train_df.values[24:, 0]

now = list()
one = list()
two = list()
wind = list()
one_wind = list()

for i, v in enumerate(train_df.values[:-2]):
    if i % 3 == 1:
        one.append((v[0] * (train_target_max - train_target_min)) + train_target_min)
        wind.append(v[1])
    elif i % 3 == 2:
        now.append(v[0] * (train_target_max - train_target_min) + train_target_min)
    else:
        two.append((v[0] * (train_target_max - train_target_min)) + train_target_min)
        one_wind.append(v[1])



data = {
    "one": one,
    "two": two,
    "wind": wind,
    "one_wind": one_wind,
    "now": now
}

_df = pd.DataFrame(data=data)

cor = _df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()