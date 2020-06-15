import pandas as pd
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_PATH, "../data/1/day_ahead")

df = pd.read_excel(os.path.join(data_path, "origin/origin.xlsx"), sheet_name=0, header=None) # single value wind power
df1 = pd.read_excel(os.path.join(data_path, "origin/origin.xlsx"), sheet_name=1, header=None) # nwp wind power
df2 = pd.read_excel(os.path.join(data_path, "origin/origin.xlsx"), sheet_name=2, header=None) # nwp wind power direction


df = df.drop(columns=[0, 1, 2, 3, 4, 5])
df1 = df1.drop(columns=[0, 1, 2, 3])
df2 = df2.drop(columns=[0, 1, 2, 3])

df[6] = df[6] ** 3
df[6] = (df[6] - df[6].min()) / (df[6].max() - df[6].min())

df = df.iloc[::3, :].reset_index(drop=True)

for name in df1.columns:
    df1[name] = df1[name] ** 3
    df1[name] = (df1[name] - df1[name].min()) / (df1[name].max() - df1[name].min())

for name in df2.columns:
    df2[name] = (df2[name] - df2[name].min()) / (df2[name].max() - df2[name].min())

df3 = pd.read_excel(os.path.join(data_path, "origin/train_in.xlsx"), header=None)
df3 = pd.concat([df3, df[13:2221].reset_index(drop=True)], axis=1, ignore_index=True)
# df3 = pd.concat([df3, df1[13:2221].reset_index(drop=True)], axis=1, ignore_index=True)
df3 = pd.concat([df3, df2[13:2221].reset_index(drop=True)], axis=1, ignore_index=True)

df4 = pd.read_excel(os.path.join(data_path, "origin/test_in.xlsx"), header=None)
df4 = pd.concat([df4, df[2229:2421].reset_index(drop=True)], axis=1, ignore_index=True)
# df4 = pd.concat([df4, df1[2229:2421].reset_index(drop=True)], axis=1, ignore_index=True)
df4 = pd.concat([df4, df2[2229:2421].reset_index(drop=True)], axis=1, ignore_index=True)

df3.to_excel(os.path.join(data_path, "train_in.xlsx"), index=False, header=False)
df4.to_excel(os.path.join(data_path, "test_in.xlsx"), index=False, header=False)
