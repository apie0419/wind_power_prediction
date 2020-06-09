import pandas as pd
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_PATH, "../data/1/day_ahead")

df = pd.read_excel(os.path.join(data_path, "origin/origin.xlsx"), sheet_name=0, header=None)
df1 = pd.read_excel(os.path.join(data_path, "origin/orgin.xlsx"), sheet_name=2, header=None)

df = df.drop(columns=[0, 1, 2, 3, 4, 5])
df1 = df1.drop(columns=[0, 1, 2, 3])

df[6] = df[6] ** 3
df[6] = (df[6] - df[6].min()) / (df[6].max() - df[6].min())
df = df.iloc[::3, :].reset_index(drop=True)


for name in df1.columns:
    df1[name] = df1[name] ** 3
    df1[name] = (df1[name] - df1[name].min()) / (df1[name].max() - df1[name].min())

df2 = pd.read_excel(os.path.join(data_path, "origin/train_in.xlsx"), header=None)

df2 = pd.concat([df2, df[:2208].reset_index(drop=True)], axis=1, ignore_index=True)
df2 = pd.concat([df2, df1[13:2221].reset_index(drop=True)], axis=1, ignore_index=True)

df3 = pd.read_excel(os.path.join(data_path, "origin/test_in.xlsx"), header=None)
df3 = pd.concat([df3, df[2216:2408].reset_index(drop=True)], axis=1, ignore_index=True)
df3 = pd.concat([df3, df1[2229:2421].reset_index(drop=True)], axis=1, ignore_index=True)

df2.to_excel(os.path.join(data_path, "train_in.xlsx"), index=False, header=False)
df3.to_excel(os.path.join(data_path, "test_in.xlsx"), index=False, header=False)
