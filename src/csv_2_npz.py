import os
import numpy as np
import pandas as pd

DATA_PATH = "/content/uSense-tactile-data/Data sample"
data_root = "/content/tactile"

for file_name in os.listdir(DATA_PATH):
    if not file_name.endswith(".csv"):
        continue
    
    full_path = os.path.join(DATA_PATH, file_name)
    
    # parse filename: Data[Texture(003)][Speed(1200)][Force(0.0)][Trial(18)].csv
    import re
    match = re.search(r'Texture\((\d+)\).*Speed\((\d+)\).*Force\((\d+\.\d+)\).*Trial\((\d+)\)', file_name)
    if not match:
        continue
    
    texture, speed, force, trial = match.group(1), int(match.group(2)), float(match.group(3)), int(match.group(4))
    
    data = pd.read_csv(full_path, header=None).values
    
    out_dir = os.path.join(data_root, f"Texture0{texture}")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"processed_S{speed}_F{force}_T{trial}.npz")
    np.savez_compressed(save_path, data=data)
    print(f"Saved: {save_path}")

print("Done!")
