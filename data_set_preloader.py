import torch
import numpy as np
from sklearn.model_selection import train_test_split
from support_func.model_processing_iowa import EEGDatasetIowa, EEGDatasetIowa_1D,EEGDatasetIowa_1D_RGB
import support_func.wavelet_transform as wt


number_samples = 2000 #10sec
# 1. Raw loading without transform
raw_dataset = EEGDatasetIowa_1D_RGB(electrode_name= 'AFz',data_dir="./Data/iowa/IowaData.mat",T=number_samples)


print("Raw dataset Loaded, splitting")

# 2. Stratified Split BEFORE transformation
labels = np.array([raw_dataset[i]["label"] for i in range(len(raw_dataset))])
indices = np.arange(len(raw_dataset))

train_idx, val_idx = train_test_split(indices, stratify=labels, test_size=6/28, random_state=42)

print("Applying WST")
# 3. Application WST + save the train

def compute_global_absmax(dataset):
    global_abs_max = 0.0
    for i in range(len(dataset)):
        eeg_np = dataset[i]['eeg']  # shape (T,)
        cur_abs_max = np.max(np.abs(eeg_np))
        if cur_abs_max > global_abs_max:
            global_abs_max = cur_abs_max
    return global_abs_max

global_m = compute_global_absmax(raw_dataset)

transform_wst = wt.WaveletScatteringTransformTFR_RGB(T=number_samples, J=3, Q=2)

def preprocess_and_save(indices, dataset, transform, out_filename):
    tfr_list = []
    label_list = []

    for idx in indices:
        sample = dataset[idx]      # e.g. {"eeg": shape (60600,), "label": 0/1}
        out = transform(sample)    # => {"tfr": shape [scales, time'], "label": int}

        tfr_list.append(out["tfr"])
        label_list.append(out["label"])

    # Stack along dimension 0 => shape [N, scales, time']
    all_tfr = torch.stack(tfr_list, dim=0)
    all_labels = torch.tensor(label_list, dtype=torch.long)

    # Save
    torch.save({"tfr": all_tfr, "labels": all_labels}, out_filename)
    print(f"Saved {out_filename} with shape {all_tfr.shape} and labels shape {all_labels.shape}")

# 5) Process + save
preprocess_and_save(train_idx, raw_dataset, transform_wst, "train_WST_tfr_rgb.pt")
preprocess_and_save(val_idx, raw_dataset, transform_wst, "val_WST_tfr_rgb.pt")