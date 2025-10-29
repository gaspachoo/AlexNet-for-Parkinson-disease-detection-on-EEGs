import os

import h5py
import mne
import numpy as np


def load_bdf_1D(file_path, electrode_index, segment=False, segment_duration=0, T=None):
    """
    Loads a .bdf file and extracts the EEG signal for the specified electrode index.
    Segments the signal into 2-second windows.
    """
    raw = mne.io.read_raw_bdf(file_path, preload=True)
    eeg_signal = raw.get_data()[
        electrode_index, :
    ]  # Extract 1D signal for the electrode

    if T is not None:
        if len(eeg_signal) > T:
            eeg_signal = eeg_signal[:T]
        elif len(eeg_signal) < T:
            eeg_signal = np.pad(eeg_signal, (0, T - len(eeg_signal)), mode="constant")

    if segment:
        eeg_signal = segment_signal(eeg_signal, 512, segment_duration)
    return eeg_signal


def segment_signal(signal, fs, segment_duration=2):
    """
    Segments the EEG signal into non-overlapping windows of `segment_duration` seconds.
    Any remaining samples that do not fit a complete window are discarded.

    Parameters:
        signal (np.array): The 1D EEG signal.
        fs (int): Sampling frequency in Hz.
        segment_duration (int, optional): Duration of each segment in seconds (default is 2s).

    Returns:
        np.array: 2D array of shape (num_segments, segment_length).
    """
    segment_length = fs * segment_duration
    num_segments = len(signal) // segment_length
    segmented_signal = signal[: num_segments * segment_length].reshape(
        num_segments, segment_length
    )
    return segmented_signal


def load_data_iowa(data_dir, electrode_list_path, electrode_name, segment_duration):
    """
    Load EEG signals from the Iowa dataset and extract the specified electrode.
    Signals are segmented into 2-second windows (500 Hz sampling rate) and flattened.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The file {data_dir} does not exist.")
    else:
        print("Data File found, loading data...", flush=True)

    with open(electrode_list_path, "r", encoding="utf-8") as f:
        electrode_list = f.read().strip().split()

    if electrode_name not in electrode_list:
        raise ValueError(
            f"The {electrode_name} electrode is not in the file electrode_list.txt."
        )

    electrode_index = electrode_list.index(electrode_name)
    print(f"{electrode_name} found at index: {electrode_index}")

    with h5py.File(data_dir, "r") as f:
        EEG_data = f["EEG"]
        num_groups = 2  # PD and control
        num_patients = 14
        fs = 500  # Sampling frequency for Iowa dataset

        segmented_data = []
        labels = []

        ch_ref = (
            EEG_data[electrode_index][0, 0]
            if EEG_data[electrode_index].shape == (1, 1)
            else EEG_data[electrode_index][0]
        )
        if isinstance(ch_ref, h5py.Reference):
            ch_data = f[ch_ref]
        else:
            raise TypeError(f"Error: ch_ref is not a HDF5 reference but {type(ch_ref)}")

        for group_idx in range(num_groups):
            group_ref = (
                ch_data[group_idx][0]
                if ch_data[group_idx].shape == (1,)
                else ch_data[group_idx]
            )
            if isinstance(group_ref, h5py.Reference):
                group_data = f[group_ref]
            else:
                raise TypeError(
                    f"Error: group_ref is not a HDF5 reference but {type(group_ref)}"
                )

            for patient_idx in range(num_patients):
                patient_ref = (
                    group_data[patient_idx][0]
                    if group_data[patient_idx].shape == (1,)
                    else group_data[patient_idx]
                )
                if isinstance(patient_ref, h5py.Reference):
                    signal = f[patient_ref][:].squeeze()
                    segments = segment_signal(signal, fs, segment_duration)
                    segmented_data.extend(segments)  # Flatten list of segments
                    labels.extend(
                        [group_idx] * len(segments)
                    )  # Extend labels accordingly
                else:
                    raise TypeError(
                        f"Error: patient_ref is not a HDF5 reference but {type(patient_ref)}"
                    )

    print(f"{len(segmented_data)} total segments loaded.")
    return segmented_data, labels


def load_data_sandiego(
    data_dir, electrode_list_path, electrode_name, segment_duration=2
):
    """
    Loads EEG signals from the San Diego dataset, extracts specified electrode,
    and segments signals into 2-second windows (512 Hz sampling rate), flattening the output.
    """
    with open(electrode_list_path, "r", encoding="utf-8") as f:
        electrode_list = f.read().strip().split()

    if electrode_name not in electrode_list:
        raise ValueError(
            f"The {electrode_name} electrode is not in the electrode list."
        )

    electrode_index = electrode_list.index(electrode_name)
    print(f"{electrode_name} found at index: {electrode_index}")

    data = []
    labels = []
    mne.set_log_level("ERROR")
    T_max = 92160

    for sub_folder in os.listdir(data_dir):
        sub_path = os.path.join(data_dir, sub_folder)
        if not os.path.isdir(sub_path):
            continue

        if sub_folder.startswith("sub-hc"):  # Healthy control group
            label = 0
            session_path = os.path.join(sub_path, "ses-hc", "eeg")
            if os.path.exists(session_path):
                for file in os.listdir(session_path):
                    if file.endswith(".bdf"):
                        file_path = os.path.join(session_path, file)
                        segments = load_bdf_1D(
                            file_path, electrode_index, True, segment_duration, T_max
                        )
                        data.extend(segments)
                        labels.extend([label] * len(segments))

        elif sub_folder.startswith("sub-pd"):  # Parkinson's disease group
            for session, label in [("ses-off", 1), ("ses-on", 2)]:
                session_path = os.path.join(sub_path, session, "eeg")
                if os.path.exists(session_path):
                    for file in os.listdir(session_path):
                        if file.endswith(".bdf"):
                            file_path = os.path.join(session_path, file)
                            segments = load_bdf_1D(
                                file_path,
                                electrode_index,
                                True,
                                segment_duration,
                                T_max,
                            )
                            data.extend(segments)
                            labels.extend([label] * len(segments))

    print(f"{len(data)} total segments loaded.")
    return data, labels


def load_all_channels_iowa(data_dir, electrode_list_path):
    """
    Load ALL EEG channels from Iowa dataset for all subjects.
    Returns continuous recordings BEFORE segmentation to allow ICA preprocessing.

    Returns:
        all_recordings: List of dictionaries, each containing:
            - 'data': 2D numpy array (n_channels, n_samples)
            - 'label': int (0=PD, 1=Control)
            - 'subject_id': tuple (group_idx, patient_idx)
        electrode_list: List of electrode names
        fs: Sampling frequency (500 Hz)
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The file {data_dir} does not exist.")

    with open(electrode_list_path, "r", encoding="utf-8") as f:
        electrode_list = f.read().strip().split()

    fs = 500  # Iowa sampling frequency
    num_groups = 2  # PD and Control
    num_patients = 14
    all_recordings = []

    with h5py.File(data_dir, "r") as f:
        EEG_data = f["EEG"]
        n_channels = len(electrode_list)

        # Iterate through groups and patients
        for group_idx in range(num_groups):
            for patient_idx in range(num_patients):
                # Load all channels for this subject
                subject_data = []

                for electrode_index in range(n_channels):
                    ch_ref = (
                        EEG_data[electrode_index][0, 0]
                        if EEG_data[electrode_index].shape == (1, 1)
                        else EEG_data[electrode_index][0]
                    )
                    ch_data = f[ch_ref]

                    group_ref = (
                        ch_data[group_idx][0]
                        if ch_data[group_idx].shape == (1,)
                        else ch_data[group_idx]
                    )
                    group_data = f[group_ref]

                    patient_ref = (
                        group_data[patient_idx][0]
                        if group_data[patient_idx].shape == (1,)
                        else group_data[patient_idx]
                    )
                    patient_signal = f[patient_ref][:].flatten()
                    subject_data.append(patient_signal)

                # Stack all channels: shape (n_channels, n_samples)
                subject_data = np.array(subject_data)

                all_recordings.append(
                    {
                        "data": subject_data,
                        "label": group_idx,  # 0=PD, 1=Control
                        "subject_id": (group_idx, patient_idx),
                    }
                )

    print(
        f"Loaded {len(all_recordings)} subjects with {n_channels} channels each (Iowa dataset)"
    )
    return all_recordings, electrode_list, fs


def load_all_channels_sandiego(data_dir, electrode_list_path, medication=None):
    """
    Load ALL EEG channels from San Diego dataset for all subjects.
    Returns continuous recordings BEFORE segmentation to allow ICA preprocessing.

    Parameters:
        medication: Optional filter ('on', 'off', or None for both)

    Returns:
        all_recordings: List of dictionaries, each containing:
            - 'data': 2D numpy array (n_channels, n_samples)
            - 'label': int (0=HC, 1=PD_OFF, 2=PD_ON or 0=HC, 1=PD if medication specified)
            - 'subject_id': str (e.g., 'hc1', 'pd11')
            - 'session': str (e.g., 'hc', 'off', 'on')
        electrode_list: List of electrode names
        fs: Sampling frequency (512 Hz)
    """
    with open(electrode_list_path, "r", encoding="utf-8") as f:
        electrode_list = f.read().strip().split()

    fs = 512
    all_recordings = []
    mne.set_log_level("ERROR")
    T_max = 92160  # Maximum length for consistency

    for sub_folder in os.listdir(data_dir):
        sub_path = os.path.join(data_dir, sub_folder)
        if not os.path.isdir(sub_path):
            continue

        if sub_folder.startswith("sub-hc"):  # Healthy control
            subject_id = sub_folder.replace("sub-", "")
            session = "hc"
            label = 0

            session_path = os.path.join(sub_path, "ses-hc", "eeg")
            if os.path.exists(session_path):
                for file in os.listdir(session_path):
                    if file.endswith(".bdf"):  # San Diego uses .bdf files
                        file_path = os.path.join(session_path, file)
                        raw = mne.io.read_raw_bdf(
                            file_path, preload=True, verbose=False
                        )
                        data = raw.get_data()  # Shape: (n_channels, n_samples)

                        # Truncate to T_max if necessary
                        if data.shape[1] > T_max:
                            data = data[:, :T_max]

                        all_recordings.append(
                            {
                                "data": data,
                                "label": label,
                                "subject_id": subject_id,
                                "session": session,
                            }
                        )

        elif sub_folder.startswith("sub-pd"):  # Parkinson's disease
            subject_id = sub_folder.replace("sub-", "")

            for session, raw_label in [("off", 1), ("on", 2)]:
                # Skip if medication filter is specified
                if medication is not None and session != medication:
                    continue

                session_path = os.path.join(sub_path, f"ses-{session}", "eeg")
                if os.path.exists(session_path):
                    for file in os.listdir(session_path):
                        if file.endswith(".bdf"):  # San Diego uses .bdf files
                            file_path = os.path.join(session_path, file)
                            raw = mne.io.read_raw_bdf(
                                file_path, preload=True, verbose=False
                            )
                            data = raw.get_data()  # Shape: (n_channels, n_samples)

                            # Truncate to T_max if necessary
                            if data.shape[1] > T_max:
                                data = data[:, :T_max]

                            # Adjust label if medication is specified
                            label = 1 if medication is not None else raw_label

                            all_recordings.append(
                                {
                                    "data": data,
                                    "label": label,
                                    "subject_id": subject_id,
                                    "session": session,
                                }
                            )

    print(
        f"Loaded {len(all_recordings)} recordings with {len(electrode_list)} channels each (San Diego dataset)"
    )
    return all_recordings, electrode_list, fs
