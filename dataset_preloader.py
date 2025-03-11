import torch
import numpy as np
from sklearn.model_selection import train_test_split
from support_func.dataset_class import *
import support_func.wavelet_transform as wt


def compute_global_absmax(dataset):
    global_abs_max = 0.0
    for i in range(len(dataset)):
        eeg_np = dataset[i]['eeg']  # shape (T,)
        cur_abs_max = np.max(np.abs(eeg_np))
        if cur_abs_max > global_abs_max:
            global_abs_max = cur_abs_max
    return global_abs_max

def process_and_save(indices, dataset,transform):
    images_list = []
    labels_list = []

    for idx in indices:
        sample = dataset[idx]      # e.g. {"eeg": shape (60600,), "label": 0/1}
        out = transform(sample)

        images_list.append(out["image"])
        labels_list.append(out["label"])

    # Stack along dimension 0 => shape [N, scales, time']
    all_images = torch.stack(images_list, dim=0)
    all_labels = torch.tensor(labels_list, dtype=torch.long)
    
    return all_images,all_labels

def preload_dataset(folder_path,electrode_name,electrode_list_path,number_samples,save_path= None, medication=None):
    # 1. Raw loading without transform
    raw_dataset = EEGDataset_1D(folder_path,electrode_name, electrode_list_path, number_samples, medication)

    print("Raw dataset Loaded, splitting")

    # 2. Stratified Split BEFORE transformation
    labels = np.array([raw_dataset[i]["label"] for i in range(len(raw_dataset))])
    indices = np.arange(len(raw_dataset))

    train_idx, val_idx = train_test_split(indices, stratify=labels, test_size=0.2, random_state=42)

    print("Applying WST")
    
    # 3. Wavelest Scattering Transform
    #global_m = compute_global_absmax(raw_dataset) -> global normalization could be added ad an argument in WST
    transform_wst = wt.WaveletScatteringTransform(number_samples, J=3, Q=2)

    all_images,all_labels = process_and_save(train_idx,raw_dataset,transform_wst)
    dataset = list(zip(all_images,all_labels))
    
    if save_path != None:
        torch.save(dataset, save_path)
        print(f"Saved {save_path} with shape {all_images.shape} and labels shape {all_labels.shape}")

    return dataset


if __name__ == "__main__":
    number_samples = 2048 #max for iowa : 60600, max for san_diego : 92160
    folder_path = "./Data/iowa/IowaData.mat"
    electrode_name='AF4'
    electrode_list_path="./Data/san_diego/electrode_list.txt"
    medication = 'on'
    train = preload_dataset(folder_path,electrode_name,electrode_list_path,number_samples,"./Datasets_pt/train_iowa.pt", medication)
    validate = preload_dataset(folder_path,electrode_name,electrode_list_path,number_samples, "./Datasets_pt/val_iowa.pt",medication)

