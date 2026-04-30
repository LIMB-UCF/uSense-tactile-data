import os
import numpy as np
import random

# ==== Config ====
TYPES = [0, 1, 2, 3]
DENSITIES = [1, 2, 3, 4, 5]

SPEEDS = [1200, 2400, 3600, 4800, 6000] 
FORCES = [0.0, 0.5, 1.0, 1.5]
SAMPLING_RATE = 338
SPEED_TIME_RANGES = {1200: (18.5, 19.5),
                    2400: (18.0, 19.0),
                    3600: (17.8, 18.8),
                    4800: (17.7, 18.7),
                    6000: (17.5, 18.5),
                }

texture_ids = [f"{t}{d}" for t in TYPES for d in DENSITIES]
texture_ids.append("00")
print(f' total texture ids: {len(texture_ids)} -> {texture_ids}')


def load_signal1(file_path, speed):
    try:
        with np.load(file_path, allow_pickle=True) as data:
            #print(f"\n--- Loading: {file_path}")
            #print(f"Keys in file: {list(data.keys())}")
            raw = data['data']
            #print(f"Raw shape: {raw.shape}, dtype: {raw.dtype}")
            signal = raw[:, :9]
          
        start_sec = SPEED_TIME_RANGES[speed][0]
        start_idx = int(start_sec * SAMPLING_RATE)
        end_idx = int((start_sec + 1) * SAMPLING_RATE)
        
        if end_idx > signal.shape[0]:
            print(f"Skipping: signal too short ({signal.shape[0]})")
            return None

        signal = signal[start_idx:end_idx, :]
        signal = np.array(signal, dtype=np.float32)
        #print(f"Min: {signal.min():.4f}, Max: {signal.max():.4f}, Mean: {signal.mean():.4f}")
        return signal

    except Exception as e:
        print(f"Loading failed for {file_path}: {e}")
        return None

def load_signal(file_path, speed):
    try:
        with np.load(file_path, allow_pickle=True) as data:
            raw = data['data']  # shape: (N, 11)

        # --- Outlier removal (load cell = col 9, 0-indexed) ---
        load_cell = raw[:, 9]
        if np.any(np.abs(load_cell) > OUTLIER_THRESHOLD):
            print(f"  Outlier detected, discarding: {file_path}")
            return None

        # --- Baseline correction: subtract per-channel mean of first 10s (taxels only, cols 0-8) ---
        baseline_end = int(BASELINE_SECONDS * SAMPLING_RATE)  # 10 * 338 = 3380 samples
        baseline_mean = raw[:baseline_end, :9].mean(axis=0)   
        raw[:, :9] -= baseline_mean                          

        signal = raw[:, :9]
        start_sec = SPEED_TIME_RANGES[speed][0]
        start_idx = int(start_sec * SAMPLING_RATE)
        end_idx = int((start_sec + 1) * SAMPLING_RATE)

        if end_idx > signal.shape[0]:
            print(f"  Skipping: signal too short ({signal.shape[0]} samples)")
            return None

        return signal[start_idx:end_idx, :].astype(np.float32)

    except Exception as e:
        print(f"  Load failed for {file_path}: {e}")
        return None

BASE_PATH = "/content/tactile"
OUTLIER_THRESHOLD = 250_000
BASELINE_SECONDS = 10

if __name__ == "__main__":
    textures_data = []
    labels = []
    save_dir = "/content/output/segmented_data"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "segmented_data.npz")

    open(save_path, "w").close() if os.path.exists(save_path) else print(f"Creating {save_path}")

    for texture in texture_ids:
        for speed in SPEEDS:
            for force in FORCES:
                folder = f'Texture0{texture.zfill(3)}'
                label = f'T{texture}_S{speed}_F{force}'

                # --- Pass 1: collect valid trials ---
                valid_signals = {}
                bad_trials = []
                for trial in range(1, 101):
                    file = f'processed_S{speed}_F{force}_T{trial}.npz'
                    path = os.path.join(BASE_PATH, folder, file)
                    if not os.path.isfile(path) or os.path.getsize(path) == 0:
                        continue
                    signal = load_signal(path, speed)
                    if signal is not None:
                        valid_signals[trial] = signal
                    else:
                        bad_trials.append(trial)  # outlier or too short

                if not valid_signals:
                    print(f"  No valid trials for {label}, skipping.")
                    continue

                # --- Pass 2: replace bad trials with random valid one ---
                all_signals = dict(valid_signals)  # start with valid ones
                for trial in bad_trials:
                    replacement = valid_signals[random.choice(list(valid_signals.keys()))].copy()
                    all_signals[trial] = replacement
                    print(f"  Replaced bad trial T{trial} in {label}")

                # --- Segment all trials ---
                for signal in all_signals.values():
                    for start in range(0, signal.shape[0] - 32 + 1, 32):
                        segment = signal[start:start + 32]
                        textures_data.append(segment)
                        labels.append(label)

    # --- Save ---
    if textures_data:
        textures_data = np.array(textures_data)
        textures_data = textures_data.transpose(0, 2, 1)
        textures_data = textures_data.reshape(-1, 32)
        labels = np.repeat(labels, 9)
        labels = np.array(labels)
        np.savez_compressed(save_path, data=textures_data, labels=labels)
        print(f"Saved {save_path} | data: {textures_data.shape}, labels: {labels.shape}")
    else:
        print("No data found.")
