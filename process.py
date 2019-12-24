import pandas as pd
import os
from datetime import datetime, timedelta

base_path = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(base_path, "Renew")

output_path = os.path.join(base_path, "Output")

if not os.path.exists(output_path):
    os.mkdir(output_path)
if not os.path.exists(os.path.join(output_path, "Wind")):
    os.mkdir(os.path.join(output_path, "Wind"))
if not os.path.exists(os.path.join(output_path, "Solar")):
    os.mkdir(os.path.join(output_path, "Solar"))

def time_process(dt):
    if dt.minute % 5 >= 3:
        dt += timedelta(minutes=5-(dt.minute % 5))
    elif dt.minute % 5 < 3 and dt.minute % 5 > 0:
        dt -= timedelta(minutes=dt.minute % 5)
    return dt

for file in os.listdir(os.path.join(data_path, "Wind")):
    f = os.path.join(data_path, "Wind", file)
    df = pd.read_csv(f)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["datetime"] = df["datetime"].apply(time_process)
    df.to_csv(os.path.join(output_path, "Wind", file), index=False)
    print (f"Finish: {file}")

for file in os.listdir(os.path.join(data_path, "Solar")):
    f = os.path.join(data_path, "Solar", file)
    df = pd.read_csv(f)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["datetime"] = df["datetime"].apply(time_process)
    df.to_csv(os.path.join(output_path, "Solar", file), index=False)
    print (f"Finish: {file}")