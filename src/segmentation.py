import os
import numpy as np
import re

# ==== Config ====
BASE_PATH = "/content/tactile"
SAMPLING_RATE = 338

SPEED_TIME_RANGES = {
    1200: (18.5, 19.5),
    2400: (18.0, 19.0),
    3600: (17.8, 18.8),
    4800: (17.7, 18.7),
    6000: (17.5, 18.5),
}

SEG_LEN = 32

def parse_filename(file_name):
    match = re.search(r'S(\d+)_F([\d\.]+)_T(\d+)', file_name)
    if match:
        speed = int(match.group(1))
        force = float(match.group(2))
        trial = int(match.group(3))
        return speed, force, trial
    return None

def load_signal(file_path, speed):
    try:
        with np.load(file_path, allow_pickle=True) as data:
            signal = data['data']

        # Ensure correct shape
        if signal.shape[1] < 9:
            return None

        signal = signal[:, :9]

        # Crop using speed-specific window
        if speed not in SPEED_TIME_RANGES:
            return None

        start_sec = SPEED_TIME_RANGES[speed][0]
        start_idx = int(start_sec * SAMPLING_RATE)
        end_idx = start_idx + SAMPLING_RATE  # exactly 1 second

        if end_idx > signal.shape[0]:
            return None

        return signal[start_idx:end_idx].astype(np.float32)

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


# ==== Main ====

textures_data = []
labels = []

save_dir = "/content/output/segmented_data"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "segmented_data.npz")

print(f"Scanning folders in {BASE_PATH}...")

for folder in os.listdir(BASE_PATH):
    folder_path = os.path.join(BASE_PATH, folder)

    if not os.path.isdir(folder_path):
        continue

    # Example: Texture0003 → extract 003
    texture_match = re.search(r'Texture0*(\d+)', folder)
    if not texture_match:
        continue

    texture_id = texture_match.group(1)

    print(f"Processing {folder}...")

    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".npz"):
            continue

        parsed = parse_filename(file_name)
        if parsed is None:
            continue

        speed, force, trial = parsed

        file_path = os.path.join(folder_path, file_name)

        signal = load_signal(file_path, speed)

        if signal is None:
            continue

        # Segment properly
        num_segments = signal.shape[0] // SEG_LEN

        for i in range(num_segments):
            segment = signal[i*SEG_LEN:(i+1)*SEG_LEN]  # (32, 9)

            textures_data.append(segment)
            labels.append(f"T{texture_id}_S{speed}_F{force}")

# Convert to array
textures_data = np.array(textures_data)  # (N, 32, 9)
labels = np.array(labels)

print(f"Before reshape: {textures_data.shape}, labels: {labels.shape}")

# OPTIONAL: only if you intentionally want channel-wise samples
textures_data = textures_data.transpose(0, 2, 1)  # (N, 9, 32)
textures_data = textures_data.reshape(-1, 32)
labels = np.repeat(labels, 9)

print(f"Final shape: {textures_data.shape}, labels: {labels.shape}")

np.savez_compressed(save_path, data=textures_data, labels=labels)

print(f"Saved to {save_path}")
