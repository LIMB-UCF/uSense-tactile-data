import os
import numpy as np
import pandas as pd
import re

DATA_PATH  = "/content/uSense-tactile-data/Data sample"
data_root  = "/content/tactile"
DEBUG      = False

# Preprocessing parameters
OUTLIER_THRESH = 250_000
CONTACT_THRESH = 2_000
DUR_REQUIRED   = 24.9
WINDOW_KEEP_S  = (10.0, 25.0)
TOL            = 1e-2
SENSOR_COLS    = list(range(10))
TIME_COL       = 10

written = skipped = 0

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

        df = pd.read_csv(full_path, header=None)
        if df.shape[1] < 11:
            #print(f"Bad column count, skipping: {file_name}")
            skipped += 1
            continue

        sensors = df.iloc[:, SENSOR_COLS]
        time    = df.iloc[:, TIME_COL].astype(float)

        # Remove load cell outliers
        ok = sensors.iloc[:, 9] <= OUTLIER_THRESH
        sensors, time = sensors[ok].reset_index(drop=True), time[ok].reset_index(drop=True)
      
        drop_idx = np.where(np.diff(time) < 0)[0]
        if drop_idx.size:
            cut = drop_idx[0] + 1
            sensors, time = sensors.iloc[:cut], time.iloc[:cut]

        # Convert to relative time and keep first 25s
        t_rel  = time - time.iloc[0]
        keep25 = t_rel < (WINDOW_KEEP_S[1] + TOL)
        sensors, t_rel = sensors[keep25].reset_index(drop=True), t_rel[keep25].reset_index(drop=True)

        if t_rel.iloc[-1] < DUR_REQUIRED:
            print(f"Skipped (too short): {file_name}")
            skipped += 1
            continue

        # Subtract baseline (avg of first 10s)
        base_mask = t_rel < WINDOW_KEEP_S[0]
        if base_mask.sum() == 0:
            print(f"Skipped (offset failure): {file_name}")
            skipped += 1
            continue
        offsets     = sensors[base_mask].mean()
        sensors_off = sensors - offsets

        # Contact flag (on full 25s)
        contact_flag = (sensors_off.iloc[:, 9] >= CONTACT_THRESH).astype(int)

        # Assemble final data — keep full 25s, reindex time to start at 0
        out_df = pd.concat(
            [sensors_off, t_rel.rename("time_s"), contact_flag.rename("contact_flag")],
            axis=1
        )

        if DEBUG:
            print(f"\n--- File: {file_name}")
            print(f"Texture: {texture}, Speed: {speed}, Force: {force}, Trial: {trial}")
            print(f"Shape after processing: {out_df.shape}")

        out_dir   = os.path.join(data_root, f"Texture0{texture}")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"processed_S{speed}_F{force}_T{trial}.npz")
        np.savez_compressed(save_path, data=out_df.values)
        written += 1

print(f"\nDone! Written: {written}, Skipped: {skipped}")
