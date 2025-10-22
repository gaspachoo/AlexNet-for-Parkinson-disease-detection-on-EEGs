from torch.utils.data import Dataset
import numpy as np
from support_func.import_data import load_data_iowa, load_data_sandiego


class EEGDataset_1D(Dataset):
    def __init__(
        self,
        data_dir,
        electrode_name,
        electrode_list_path,
        segment_duration,
        medication=None,
    ):
        """
        Initialize EEGDataset_1D

        Parameters
        ----------
        data_dir : str
            Path to the dataset
        electrode_name : str
            Name of the electrode
        electrode_list_path : str
            Path to the file containing the list of electrodes
        T : int
            Length of each segment
        medication : str, optional
            Medication status for San Diego dataset. If provided, the dataset will be filtered to include only the specified medication status.

        Returns
        -------
        None
        """
        super().__init__()
        self.segment_duration = segment_duration
        self.medication = medication

        self.fs = 500.0 if "iowa" in data_dir.lower() else 512.0
        self.sample_length = int(self.fs * self.segment_duration)  # automatique

        if "iowa" in data_dir.lower():
            self.data_source = "iowa"
            self.eeg_data, self.labels = load_data_iowa(
                data_dir, electrode_list_path, electrode_name, segment_duration
            )
        else:
            self.data_source = "san_diego"
            self.eeg_data, self.labels = load_data_sandiego(
                data_dir, electrode_list_path, electrode_name, segment_duration
            )

            if medication is not None:
                selected_label = (
                    2 if medication.lower() == "on" else 1
                )  # 2 = ON, 1 = OFF
                filtered_data, filtered_labels = [], []

                for i in range(len(self.labels)):
                    if self.labels[i] == 0 or self.labels[i] == selected_label:
                        filtered_data.append(self.eeg_data[i])
                        filtered_labels.append(
                            1 if self.labels[i] == 2 else self.labels[i]
                        )  # Convert 2 → 1

                if len(filtered_data) == 0:
                    raise ValueError(
                        f"No samples found for HC + PD {medication.upper()} in San Diego dataset!"
                    )

                self.eeg_data = np.array(filtered_data, dtype=object)
                self.labels = np.array(filtered_labels)  # Ensure labels are updated

                print(
                    f"Dataset filtered: {len(filtered_data)} samples (HC + PD {medication.upper()})"
                )

        assert len(self.eeg_data) == len(self.labels), (
            "Mismatch between EEG data and labels."
        )

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg_1d = self.eeg_data[idx]

        if len(eeg_1d) > self.sample_length:
            eeg_1d = eeg_1d[: self.sample_length]
        elif len(eeg_1d) < self.sample_length:
            eeg_1d = np.pad(
                eeg_1d, (0, self.sample_length - len(eeg_1d)), mode="constant"
            )

        eeg_1d = eeg_1d.astype(np.float32).reshape(-1, 1)
        sample = {"eeg": eeg_1d, "label": int(self.labels[idx])}
        return sample
