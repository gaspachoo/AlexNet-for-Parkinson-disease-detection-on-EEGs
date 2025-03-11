import torch
import numpy as np
from sklearn.model_selection import train_test_split
from support_func.dataset_class import *
import support_func.wavelet_transform as wt


number_samples = 90000
# 1. Raw loading without transform
raw_dataset = EEGDataset_1D("./Data/san_diego",electrode_name='AF4', electrode_list_path="./Data/san_diego/electrode_list.txt", medication='on', T=number_samples)

print("Raw dataset Loaded, splitting")

# 2. Stratified Split BEFORE transformation
labels = np.array([raw_dataset[i]["label"] for i in range(len(raw_dataset))])
indices = np.arange(len(raw_dataset))

train_idx, val_idx = train_test_split(indices, stratify=labels, test_size=0.2, random_state=42)

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

transform_wst = wt.WaveletScatteringTransform(T=number_samples, J=3, Q=2)

def preprocess_and_save(indices, dataset, transform, out_filename):
    images_list = []
    labels_list = []

    for idx in indices:
        sample = dataset[idx]      # e.g. {"eeg": shape (60600,), "label": 0/1}
        out = transform(sample)    # => {"tfr": shape [scales, time'], "label": int}

        images_list.append(out["image"])
        labels_list.append(out["label"])

    # Stack along dimension 0 => shape [N, scales, time']
    all_images = torch.stack(images_list, dim=0)
    all_labels = torch.tensor(labels_list, dtype=torch.long)

    # Save
    #torch.save({"images": all_images, "labels": all_labels}, out_filename)
    torch.save(list(zip(all_images,all_labels)), out_filename)

    print(f"Saved {out_filename} with shape {all_images.shape} and labels shape {all_labels.shape}")

# 5) Process + save
preprocess_and_save(train_idx, raw_dataset, transform_wst, "./Datasets_pt/train_sd_on.pt")
preprocess_and_save(val_idx, raw_dataset, transform_wst, "./Datasets_pt/val_sd_on.pt")