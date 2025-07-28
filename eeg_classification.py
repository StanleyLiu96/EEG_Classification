import os
import numpy as np
import mne
from braindecode.datasets.base import BaseDataset, BaseConcatDataset

# Path to folder of EEG .npy files
eeg_dir = "/users/PAS2301/liu215229932/Music_Project/Dataset/MADEEG/processed_data/response_npy"

# Sampling rate and channel names
sfreq = 256  # Hz
n_channels = 20
ch_names = [f"EEG {i+1}" for i in range(n_channels)]
ch_types = ["eeg"] * n_channels

# Build mapping of instrument labels to int class indices
def extract_label_from_filename(filename):
    return filename.split("_")[-2]

all_files = sorted([f for f in os.listdir(eeg_dir) if f.endswith("_response.npy")])
all_labels = sorted({extract_label_from_filename(f) for f in all_files})
label_to_index = {label: i for i, label in enumerate(all_labels)}

# Build dataset list
braindecode_datasets = []

for fname in all_files:
    path = os.path.join(eeg_dir, fname)

    # Load EEG numpy file: shape (20, 4864)
    data = np.load(path)

    # Convert to RawArray
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    # Get label and convert to int
    label_str = extract_label_from_filename(fname)
    label_idx = label_to_index[label_str]

    # Wrap into Braindecode BaseDataset
    desc = {"subject": 0, "target": label_idx, "label_str": label_str}
    base_dataset = BaseDataset(raw, description=desc)
    braindecode_datasets.append(base_dataset)

# Combine into one big dataset
dataset = BaseConcatDataset(braindecode_datasets)

print(f"✅ Created dataset with {len(dataset)} trials and labels: {label_to_index}")
