from torch.utils.data import Dataset
import numpy as np
from support_func.import_data import *
from support_func.filters import butter_bandpass_filter

class EEGDataset_1D(Dataset):
    def __init__(self, data_dir, electrode_name, electrode_list_path, T, medication=None):
        super().__init__()
        self.T = T
        self.medication = medication

        if "iowa" in data_dir.lower():
            self.data_source = "iowa"
            self.eeg_data, self.labels = load_data_iowa_1D_seg(data_dir, electrode_list_path, electrode_name)
        
        else:
            self.data_source = "san_diego"
            self.eeg_data, self.labels = load_data_sandiego_1D_seg(data_dir, electrode_list_path, electrode_name)

            if medication is not None:
                selected_label = 2 if medication.lower() == "on" else 1  # 2 = ON, 1 = OFF
                filtered_data, filtered_labels = [], []

                for i in range(len(self.labels)):
                    if self.labels[i] == 0 or self.labels[i] == selected_label:  
                        filtered_data.append(self.eeg_data[i])
                        filtered_labels.append(1 if self.labels[i] == 2 else self.labels[i])  # ✅ Convert 2 → 1

                if len(filtered_data) == 0:
                    raise ValueError(f"❌ No samples found for HC + PD {medication.upper()} in San Diego dataset!")

                self.eeg_data = np.array(filtered_data, dtype=object)
                self.labels = np.array(filtered_labels)  # ✅ Ensure labels are updated

                print(f"✅ Dataset filtered: {len(filtered_data)} samples (HC + PD {medication.upper()})")
                
        assert len(self.eeg_data) == len(self.labels), "❌ Mismatch between EEG data and labels."

    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        eeg_1d = self.eeg_data[idx]  
        fs = 500.0 if self.data_source == "iowa" else 512.0
        eeg_1d = butter_bandpass_filter(eeg_1d, lowcut=0.5, highcut=40.0, fs=fs, order=4)

        if len(eeg_1d) > self.T:
            eeg_1d = eeg_1d[:self.T]  
        elif len(eeg_1d) < self.T:
            eeg_1d = np.pad(eeg_1d, (0, self.T - len(eeg_1d)), mode='constant')  

        eeg_1d = eeg_1d.astype(np.float32).reshape(-1, 1)  
        sample = {'eeg': eeg_1d, 'label': int(self.labels[idx])}  

        return sample