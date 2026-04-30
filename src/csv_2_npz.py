import os
import numpy as np
import pandas as pd
import re

#DATA_PATH = "/content/uSense-tactile-data/Data sample"
DATA_PATH = "/content/uSense-tactile-data/sample_trials" 
data_root = "/content/tactile"
DEBUG = False

for folder_name in os.listdir(DATA_PATH):
    folder_path = os.path.join(DATA_PATH, folder_name)
    if not os.path.isdir(folder_path):  
        continue

    for file_name in os.listdir(folder_path):  
        if not file_name.endswith(".csv"):
            continue

        full_path = os.path.join(folder_path, file_name)

        match = re.search(r'Texture\((\d+)\).*Speed\((\d+)\).*Force\((\d+\.\d+)\).*Trial\((\d+)\)', file_name)
        if not match:
            print(f"Could not parse: {file_name}")
            continue

        texture = match.group(1)
        speed   = int(match.group(2))
        force   = float(match.group(3))
        trial   = int(match.group(4))

        data = pd.read_csv(full_path, header=None).values
        #print(f"Loaded data shape: {data.shape} from {full_path}")

        if DEBUG:
            print(f"\n--- File: {file_name}")
            print(f"Texture: {texture}, Speed: {speed}, Force: {force}, Trial: {trial}")
            print(f"Shape: {data.shape}, dtype: {data.dtype}")
            print(f"Min: {np.min(data):.4f}, Max: {np.max(data):.4f}, Mean: {np.mean(data):.4f}")
            print(f"First 2 rows:\n{data[:2]}")
            if np.isnan(data).any():
                print("Contains NaNs")

        out_dir = os.path.join(data_root, f"Texture0{texture}")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"processed_S{speed}_F{force}_T{trial}.npz")
        np.savez_compressed(save_path, data=data)
        #print(f"Saved: {save_path}")

print("Done!")
