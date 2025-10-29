import numpy as np
from torch.utils.data import Dataset

from support_func.import_data import (
    load_all_channels_iowa,
    load_all_channels_sandiego,
    load_data_iowa,
    load_data_sandiego,
)


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


class EEGDataset_MultiChannel(Dataset):
    """
    Multi-channel EEG dataset that loads ALL channels, applies filtering (including ICA),
    then segments and extracts the target electrode.

    This enables ICA and other multi-channel preprocessing methods to be applied
    on continuous recordings BEFORE segmentation.
    """

    def __init__(
        self,
        data_dir,
        electrode_name,
        electrode_list_path,
        segment_duration,
        filter_func=None,
        medication=None,
    ):
        """
        Initialize EEGDataset_MultiChannel

        Parameters:
            data_dir: Path to the dataset
            electrode_name: Name of the target electrode to extract after filtering
            electrode_list_path: Path to electrode_list.txt
            segment_duration: Segment duration in seconds
            filter_func: Optional filtering function to apply to multi-channel data.
                        Should accept (data, fs) and return filtered data of same shape.
                        Examples: MNE_ICA_Wavelet, SKLFast_ICA
            medication: Medication status for San Diego dataset ('on', 'off', or None)
        """
        super().__init__()
        self.segment_duration = segment_duration
        self.medication = medication
        self.electrode_name = electrode_name
        self.filter_func = filter_func

        # Determine dataset source
        self.data_source = "iowa" if "iowa" in data_dir.lower() else "san_diego"
        self.fs = 500.0 if self.data_source == "iowa" else 512.0
        self.sample_length = int(self.fs * self.segment_duration)

        # Load all channels for all subjects
        print(f"Loading multi-channel data from {self.data_source}...")
        if self.data_source == "iowa":
            recordings, electrode_list, fs = load_all_channels_iowa(
                data_dir, electrode_list_path
            )
        else:
            recordings, electrode_list, fs = load_all_channels_sandiego(
                data_dir, electrode_list_path, medication
            )

        if electrode_name not in electrode_list:
            raise ValueError(f"Electrode {electrode_name} not found in electrode list.")

        self.target_channel_idx = electrode_list.index(electrode_name)
        print(f"Target electrode: {electrode_name} (index {self.target_channel_idx})")

        # Process each recording: apply filter, then segment
        self.eeg_data = []
        self.labels = []

        for i, recording in enumerate(recordings):
            multi_channel_data = recording["data"]  # Shape: (n_channels, n_samples)
            label = recording["label"]

            # Apply filtering function if provided (e.g., ICA)
            if filter_func is not None:
                print(
                    f"  Applying filter to recording {i + 1}/{len(recordings)}...",
                    end="\r",
                )
                try:
                    multi_channel_data = filter_func(multi_channel_data, sfreq=int(fs))
                except Exception as e:
                    print(
                        f"\n  Warning: Filter failed for recording {i + 1}: {e}. Using unfiltered data."
                    )

            # Extract target channel
            target_signal = multi_channel_data[self.target_channel_idx, :]

            # Segment the signal
            segment_length = int(fs * segment_duration)
            num_segments = len(target_signal) // segment_length

            for seg_idx in range(num_segments):
                start = seg_idx * segment_length
                end = start + segment_length
                segment = target_signal[start:end]
                self.eeg_data.append(segment)
                self.labels.append(label)

        print(f"\nTotal segments created: {len(self.eeg_data)}")
        self.eeg_data = np.array(self.eeg_data, dtype=object)
        self.labels = np.array(self.labels)

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
