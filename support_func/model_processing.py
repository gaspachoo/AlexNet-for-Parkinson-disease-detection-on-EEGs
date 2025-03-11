from support_func import import_data
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt

def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=500.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)
    return y

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

class EEGDatasetSanDiego_1D(Dataset):
    """
    PyTorch Dataset for San Diego EEG, extracting only a specific electrode.
    - data_dir: Path to the directory containing subject folders.
    - electrode_name: The name of the electrode to extract.
    - electrode_list_path: Path to the file containing electrode names.
    - transform: Optional transformation (e.g., wavelet scattering).
    - T: Target length for EEG signals (default is 60600).
    """
    def __init__(self, data_dir, electrode_name, electrode_list_path="./Data/san_diego/electrode_list.txt", transform=None, T=60600):
        super().__init__()
        self.T = T
        self.eeg_data, self.labels = import_data.load_data_sandiego_1D(data_dir, electrode_list_path, electrode_name)  # Load only this electrode
        self.transform = transform

        # Ensure consistency
        assert len(self.eeg_data) == len(self.labels), "❌ Mismatch between EEG data and labels."

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        """
        Returns {'eeg': (T, 1), 'label': int} and applies self.transform if provided.
        """
        eeg_1d = self.eeg_data[idx]  # Shape (variable T,)

        # Apply bandpass filtering
        eeg_1d = butter_bandpass_filter(eeg_1d, lowcut=0.5, highcut=40.0, fs=512.0, order=4)

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
    
def EEGDatasetSanDiego_Medication(data_dir, electrode_name, medication, electrode_list_path="./Data/san_diego/electrode_list.txt", transform=None, T=60600):
    """
    PyTorch Dataset for San Diego EEG with medication filtering.
    - medication: Boolean (True for ON, False for OFF).
    - Includes only HC and PD (ON or OFF based on medication argument).
    """
    selected_label = 2 if medication else 1  # 2 for ON, 1 for OFF
    
    eeg_data, labels = import_data.load_data_sandiego_1D(data_dir, electrode_list_path, electrode_name)
    
    filtered_data = []
    filtered_labels = []
    
    for i in range(len(labels)):
        if labels[i] == 0 or labels[i] == selected_label:  # Keep HC and selected PD group
            filtered_data.append(eeg_data[i])
            filtered_labels.append(labels[i])
    
    dataset = EEGDatasetSanDiego_1D(data_dir, electrode_name, electrode_list_path, transform, T)
    dataset.eeg_data = np.array(filtered_data, dtype=object)
    dataset.labels = np.array(filtered_labels)
    
    print(f"✅ Dataset created with {len(filtered_data)} samples (HC + PD {'ON' if medication else 'OFF'})")
    return dataset
    
    