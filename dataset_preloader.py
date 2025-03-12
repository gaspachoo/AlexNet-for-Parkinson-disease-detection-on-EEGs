import torch
import numpy as np
from sklearn.model_selection import train_test_split
from support_func.dataset_class import *
import support_func.wavelet_transform as wt


def global_norm(images_tensor):
    """Apply global normalization across all images in a tensor."""
    if not isinstance(images_tensor, torch.Tensor):
        raise TypeError(f"Expected a tensor, but got {type(images_tensor)}")

    # Compute global min/max from the tensor itself
    global_min = images_tensor.min()
    global_max = images_tensor.max()

    # Normalize all images based on global min/max
    return (images_tensor - global_min) / (global_max - global_min + 1e-8)


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

def preload_dataset(mode,electrode_name,number_samples,save=False, medication=None):
    
    # 0. Get folder path, electrode_list_path
    if mode=="iowa":
        assert(medication==None), "Medication setting should be None for Iowa Dataset"
        folder_path = "./Data/iowa/IowaData.mat"
        electrode_list_path="./Data/iowa/electrode_list.txt"
    else:
        folder_path = "./Data/san_diego"
        electrode_list_path="./Data/san_diego/electrode_list.txt"
    
    # 1. Raw loading without transform
    raw_dataset = EEGDataset_1D(folder_path,electrode_name, electrode_list_path, number_samples, medication)

    print("Raw dataset Loaded, splitting")

    # 2. Stratified Split BEFORE transformation
    labels = np.array([raw_dataset[i]["label"] for i in range(len(raw_dataset))])
    indices = np.arange(len(raw_dataset))

    train_idx, val_idx = train_test_split(indices, stratify=labels, test_size=0.2, random_state=42) ## Select here the test size

    print("Applying WST")
    
    # 3. Wavelest Scattering Transform
    
    transform_wst = wt.WaveletScatteringTransform(number_samples, J=8, Q=24)

    train_images,train_labels = process_and_save(train_idx,raw_dataset,transform_wst)
    val_images, val_labels = process_and_save(val_idx,raw_dataset,transform_wst)
    
    #train_images = global_norm(train_images) #Apply global norm
    #val_images = global_norm(val_images)
    
    train_dataset = list(zip(train_images,train_labels))
    val_dataset = list(zip(val_images,val_labels))
    
    if save == True:
        
        #Define paths
        path = './Datasets_pt/'
        if "iowa" in folder_path:
            train_path = path + 'train_iowa.pt'
            val_path = path + 'val_iowa.pt'
        else:
            if medication == None:
                train_path = path + 'train_sd_onandoff.pt'
                val_path = path + 'val_sd_onandoff.pt'
            elif medication == "on":
                train_path = path + 'train_sd_on.pt'
                val_path = path + 'val_sd_on.pt'
            else :
                train_path = path + 'train_sd_off.pt'
                val_path = path + 'val_sd_off.pt'
            
        torch.save(train_dataset, train_path)
        print(f"Saved {train_path} with shape {train_images.shape} and labels shape {train_labels.shape}")
        torch.save(val_dataset, val_path)
        print(f"Saved {val_path} with shape {val_images.shape} and labels shape {val_labels.shape}")

    return train_dataset,val_dataset


if __name__ == "__main__":
    number_samples = 2000 #max for iowa : 60600, max for san_diego : 92160
    electrode_name='AFz'
    medication = None # For iowa, None, for san_diego, on or off or None (not None for model training)
    mode = "iowa"
    train_set,validate_set = preload_dataset(mode,electrode_name,number_samples,True, medication)

