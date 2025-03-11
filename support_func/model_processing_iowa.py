from support_func import import_data
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt

def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=500.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)
    return y

class EEGDatasetIowa(Dataset):
    """
    Dataset PyTorch for Iowa dataset loaded as in load_data_iowa.
    - data_dir: directory of the HDF5 file.
    - transform: eventual transform (e.g. wavelet scattering) applied to each sample.
    """
    def __init__(self, data_dir, transform=None):
        super().__init__()
        # data => [28 patients], each patient => [64 channels], channel => 1D-array of shape (T,)
        data = import_data.load_data_iowa(data_dir)
        
        # labels => array de shape (28,) avec 0 ou 1
        labels = import_data.load_labels_iowa()
        
        self.eeg_data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.eeg_data)  # 28

    def __getitem__(self, idx):
        """
        Return {'eeg': (T, 64), 'label': int}.
        Then applies self.transform if a transform is given.
        """
        # Get the list of channel for the patient: 'idx' : shape = [64], each channel is (T,).
        channels_list = self.eeg_data[idx]
        channels_list = [ch[:60600] for ch in channels_list]
        
        # Empile in shape (64, T)
        eeg_2d = np.stack(channels_list, axis=0)
        
        eeg_2d = butter_bandpass_filter(eeg_2d.T, lowcut=0.5, highcut=40.0, fs=500.0, order=4).T
        eeg_2d = eeg_2d.copy()
        
        # Transpose into (T, 64) => more conventional
        eeg_2d = eeg_2d.T  # shape (T, 64)

        label = self.labels[idx]  # 0 or 1
        sample = {'eeg': eeg_2d.astype(np.float32), 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class PrecomputedEEGDataset(Dataset):
    """
    To load already trasnformed EEG and saved with .pt extension.
    """
    def __init__(self, path_to_file):
        super().__init__()
        data = torch.load(path_to_file)
        self.images = data['tfr']
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return {'image': image, 'label': label}

class EEGDatasetIowa_1D(Dataset):
    """
    PyTorch Dataset for Iowa EEG, extracting only CPz.
    - data_dir: Path to the HDF5 EEG file.
    - transform: Optional transform (e.g., wavelet scattering).
    - electrode_list_path: Path to the file containing electrode names.
    """
    def __init__(self, data_dir,electrode_name, electrode_list_path="./Data/iowa/electrode_list.txt", transform=None, T=60600):
        super().__init__()
        self.T = T
        self.eeg_data = import_data.load_data_iowa_1D(data_dir, electrode_list_path, electrode_name=electrode_name)  # Load only this electrode
        self.labels = import_data.load_labels_iowa_1D()
        self.transform = transform

        # Ensure consistency
        assert len(self.eeg_data) == len(self.labels), "❌ Mismatch between EEG data and labels."

    def __len__(self):
        return len(self.eeg_data)  # 28 patients

    def __getitem__(self, idx):
        """
        Returns {'eeg': (T, 1), 'label': int} and applies self.transform if provided.
        """
        eeg_1d = self.eeg_data[idx]  # Shape (variable T,)

        # Apply bandpass filtering
        eeg_1d = butter_bandpass_filter(eeg_1d, lowcut=0.5, highcut=40.0, fs=500.0, order=4)

        # Ensure consistent length (T=60600) - truncate or pad with zeros
        if len(eeg_1d) > self.T:
            eeg_1d = eeg_1d[:self.T]  # Truncate
        elif len(eeg_1d) < self.T:
            eeg_1d = np.pad(eeg_1d, (0, self.T - len(eeg_1d)), mode='constant')  # Pad with zeros

        eeg_1d = eeg_1d.astype(np.float32).reshape(-1, 1)  # Ensure shape (T, 1)

        sample = {'eeg': eeg_1d, 'label': self.labels[idx]}  # Package into dictionary

        if self.transform:
            sample = self.transform(sample)

        return sample
    

class EEGDatasetIowa_1D_RGB(Dataset):
    """
    PyTorch Dataset for Iowa EEG, extracting only CPz.
    - data_dir: Path to the HDF5 EEG file.
    - transform: Wavelet Scattering Transform (WST).
    - electrode_list_path: Path to the file containing electrode names.
    """
    def __init__(self, data_dir,electrode_name, electrode_list_path="./Data/iowa/electrode_list.txt", transform=None, T=60600):
        super().__init__()
        self.T = T
        self.eeg_data = import_data.load_data_iowa_1D(data_dir, electrode_list_path,electrode_name)  # Load only this electrode
        self.labels = import_data.load_labels_iowa_1D()
        self.transform = transform

        # Ensure consistency
        assert len(self.eeg_data) == len(self.labels), "❌ Mismatch between EEG data and labels."

    def __len__(self):
        return len(self.eeg_data)  # 28 patients

    def __getitem__(self, idx):
        """
        Returns {'eeg': (T, 1), 'label': int} and applies self.transform if provided.
        """
        eeg_1d = self.eeg_data[idx]  # Shape (variable T,)
        # Apply bandpass filtering
        eeg_1d = butter_bandpass_filter(eeg_1d, lowcut=0.5, highcut=40.0, fs=500.0, order=4)

        # Ensure consistent length (T=60600) - truncate or pad with zeros
        if len(eeg_1d) > self.T:
            eeg_1d = eeg_1d[:self.T]  # Truncate
        elif len(eeg_1d) < self.T:
            eeg_1d = np.pad(eeg_1d, (0, self.T - len(eeg_1d)), mode='constant')  # Pad with zeros

        eeg_1d = eeg_1d.astype(np.float32).reshape(-1, 1)  # Ensure shape (T, 1)

        sample = {'eeg': eeg_1d, 'label': self.labels[idx]}  # Package into dictionary

        if self.transform:
            sample = self.transform(sample)

        return sample
