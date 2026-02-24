
import os
import numpy as np
import pandas as pd


user = os.getenv('USERNAME') or os.getenv('USER') or 'user'
def load_texture_data(data_root, bump_type, bump_density, speed, force, trial, save_numpy=True):
    folder_name = f"Texture0{bump_type}{bump_density}"
    file_name = f"Data[Texture(0{bump_type}{bump_density})][Speed({speed})][Force({force})][Trial({trial})].csv"
    full_path = fr"C:\tactile\Data\{folder_name}\{file_name}"
    if not os.path.exists(full_path):
        return None

    data = pd.read_csv(full_path, header=None).values

    if save_numpy:
        out_dir = os.path.join(data_root, folder_name)
        os.makedirs(out_dir, exist_ok=True)  
        save_path = os.path.join(out_dir, f"processed_S{speed}_F{force}_T{trial}.npz")
        np.savez_compressed(save_path, data=data)
        return save_path
    else:
        return data


data_root = fr"C:\tactile"
# Select configuration
bump_types = [0, 1, 2, 3]
bump_densities = [0, 1, 2, 3, 4, 5]
speeds = [1200, 2400, 3600, 4800, 6000]
forces = [0.0, 0.5, 1.0, 1.5]
trials = list(range(1, 101))

for bump_type in bump_types:
    for bump_density in bump_densities:
        print(f"Processing Texture0{bump_type}{bump_density}...")
        for s in speeds:
            for f in forces:
                print(f"  Processing speed: {s}, force: {f}")
                for t in trials:
                    npz_path = load_texture_data(data_root, bump_type, bump_density, s, f, t)
                    if npz_path is not None:
                        print(f"Processed: Speed={s}, Force={f}, Trial={t} -> {npz_path}")
print("All processing complete.")

