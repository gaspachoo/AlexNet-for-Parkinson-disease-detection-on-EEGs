import torch
import numpy as np
from sklearn.model_selection import train_test_split
from support_func.dataset_class import *
import support_func.wavelet_transform as wt
from support_func.filters import *


def process_and_save(indices, dataset,transform,fs):
    images_list = []
    labels_list = []
    n_indices = len(indices)

    for i,idx in enumerate(indices):
        sample = dataset[idx]      # e.g. {"eeg": shape (60600,), "label": 0/1}
        image = sample["eeg"]
        image_filtered = bandpass_filter(image.T, lowcut=0.5, highcut=40.0, fs=fs, order=4)
        image_filtered = matlab_like_cleaning(image_filtered).T
        #image_filtered = savgol_filter(image.T, window_length=11, polyorder=3).T
        sample["eeg"] = image_filtered
        out = transform(sample)
        images_list.append(out["image"])
        labels_list.append(out["label"])
        
        av = 100*(i+1)/n_indices
        print(f'Progress : {av:.2f}%')

    # Stack along dimension 0 => shape [N, scales, time']
    all_images = torch.stack(images_list, dim=0)
    all_labels = torch.tensor(labels_list, dtype=torch.long)
    
    return all_images,all_labels

def preload_dataset(mode, electrode_name, segment_duration, save=False, medication=None):
    if mode == "iowa":
        folder_path = "./Data/iowa/IowaData.mat"
        electrode_list_path = "./Data/iowa/electrode_list.txt"
    elif mode == 'san_diego':
        folder_path = "./Data/san_diego"
        electrode_list_path = "./Data/san_diego/electrode_list.txt"
    else:
        raise ValueError("Expected 'iowa' or 'san_diego'")

    raw_dataset = EEGDataset_1D(
        folder_path, electrode_name, electrode_list_path, segment_duration, medication
    )

    print("Raw dataset loaded, splitting.")

    labels = np.array([raw_dataset[i]["label"] for i in range(len(raw_dataset))])
    indices = np.arange(len(raw_dataset))

    train_idx, val_idx = train_test_split(indices, stratify=labels, test_size=0.2, random_state=42)

    print("Applying WST")
    fs = 512 if mode == 'san_diego' else 500
    transform_wst = wt.WaveletScatteringTransform(segment_duration,fs, J=10, Q=24) #### Change WST hyperparameters	


    print("Processing and saving")
    train_images, train_labels = process_and_save(train_idx, raw_dataset, transform_wst,fs)
    val_images, val_labels = process_and_save(val_idx, raw_dataset, transform_wst,fs)

    train_dataset = list(zip(train_images, train_labels))
    val_dataset = list(zip(val_images, val_labels))

    if save:
        path = './Datasets_pt/'
        dataset_name = 'iowa' if mode == 'iowa' else f'sd_{medication or "onandoff"}'
        train_path = path + f'train_{dataset_name}_{electrode_name}.pt'
        val_path = path + f'val_{dataset_name}_{electrode_name}.pt'

        torch.save(train_dataset, train_path)
        print(f"Saved {train_path} with shape {train_images.shape} and labels shape {train_labels.shape}")
        torch.save(val_dataset, val_path)
        print(f"Saved {val_path} with shape {val_images.shape} and labels shape {val_labels.shape}")

    return train_dataset, val_dataset


if __name__ == "__main__":
    segment_duration = 5 #in seconds
    electrode_name='AFz' ## AFz for iowa, Fz for sd
    medication = None # For iowa, None, for san_diego, on or off or None (not None for model training)
    mode = "iowa" #iowa or san_diego
    train_set,validate_set = preload_dataset(mode,electrode_name,segment_duration,True, medication)

