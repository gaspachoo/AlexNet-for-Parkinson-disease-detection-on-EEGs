import numpy as np
import h5py
import os
import numpy as np
import mne

#### 1D:

def load_data_iowa_1D(data_dir, electrode_list_path, electrode_name):
    """
    Load the EEG signas and extracts only those for the electrode name given.
    """

    # Load list of electrodes
    with open(electrode_list_path, "r", encoding="utf-8") as f:
        electrode_list = f.read().strip().split()

    print("📄 Electrode list loaded")

    if electrode_name not in electrode_list:
        raise ValueError(f"⚠️ The {electrode_name} electrode is not in the file electrode_list.txt.")

    electrode_index = electrode_list.index(electrode_name)  # Find electrode index
    print(f"✅ {electrode_name} found at the index : {electrode_index}")

    with h5py.File(data_dir, 'r') as f:
        EEG_data = f['EEG']
        print("📂 Keys available in the HDF5 file :", list(f.keys()))

        num_groups = 2  # PD and control
        num_patients = 14

        electrode_data = []  # List to story only electrode's eegs


        ch_ref = EEG_data[electrode_index][0, 0] if EEG_data[electrode_index].shape == (1, 1) else EEG_data[electrode_index][0]
        print(f"🔗 Reference of {electrode_name} got :", ch_ref)

        if isinstance(ch_ref, h5py.Reference):
            ch_data = f[ch_ref]
        else:
            raise TypeError(f"❌ Error : ch_ref is not a HDF5 reference but {type(ch_ref)}")

        for group_idx in range(num_groups):
            group_ref = ch_data[group_idx][0] if ch_data[group_idx].shape == (1,) else ch_data[group_idx]
            if isinstance(group_ref, h5py.Reference):
                group_data = f[group_ref]
            else:
                raise TypeError(f"❌ Error : group_ref is not a HDF5 reference but {type(group_ref)}")

            for patient_idx in range(num_patients):
                patient_ref = group_data[patient_idx][0] if group_data[patient_idx].shape == (1,) else group_data[patient_idx]
                if isinstance(patient_ref, h5py.Reference):
                    signal = f[patient_ref][:].squeeze()  # Extract and ensure it is a vector
                    electrode_data.append(signal)
                else:
                    raise TypeError(f"❌ Erreur : patient_ref is not a HDF5 reference but {type(patient_ref)}")

    print(f"✅ Extraction finished, {len(electrode_data)} signals loaded.")
    
    labels = np.zeros(28)
    labels[:14] = 1
    return electrode_data, labels
 
def load_bdf_1D(file_path, electrode_index):
    """
    Loads a .bdf file and extracts the EEG signal for the specified electrode index.
    Segments the signal into 2-second windows.
    """
    raw = mne.io.read_raw_bdf(file_path, preload=True)
    eeg_signal = raw.get_data()[electrode_index, :]  # Extract 1D signal for the electrode
    eeg_signal = segment_signal(eeg_signal,512)
    return eeg_signal

def load_data_sandiego_1D(data_dir, electrode_list_path, electrode_name):
    """
    Loads 1D EEG signals from BDF files in `data_dir`, considering the San Diego folder structure:
    - sub-hcXX/ses-hc/eeg/*.bdf → Healthy control (HC, label=0)
    - sub-pdXX/ses-off/eeg/*.bdf → Parkinson OFF (label=1)
    - sub-pdXX/ses-on/eeg/*.bdf → Parkinson ON (label=2)
    """
    
    # Load the list of electrodes
    with open(electrode_list_path, "r", encoding="utf-8") as f:
        electrode_list = f.read().strip().split()
    
    print("📄 Electrode list loaded")
    
    if electrode_name not in electrode_list:
        raise ValueError(f"⚠️ The {electrode_name} electrode is not in the electrode list.")
    
    electrode_index = electrode_list.index(electrode_name)  # Find the index of the electrode
    print(f"✅ {electrode_name} found at index: {electrode_index}")
    
    data = []
    labels = []
    mne.set_log_level("ERROR")  # Suppress info messages, show only errors

    
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
                        signal = load_bdf_1D(file_path, electrode_index)
                        data.append(signal)
                        labels.append(label)
        
        elif sub_folder.startswith("sub-pd"):  # Parkinson's disease group
            for session, label in [("ses-off", 1), ("ses-on", 2)]:
                session_path = os.path.join(sub_path, session, "eeg")
                if os.path.exists(session_path):
                    for file in os.listdir(session_path):
                        if file.endswith(".bdf"):
                            file_path = os.path.join(session_path, file)
                            signal = load_bdf_1D(file_path, electrode_index)
                            data.append(signal)
                            labels.append(label)
    
    data = np.array(data, dtype=object)
    labels = np.array(labels)
    print(f"✅ {len(data)} signals loaded.")
    return data, labels


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
    segmented_signal = signal[:num_segments * segment_length].reshape(num_segments, segment_length)
    return segmented_signal


def load_data_iowa_1D_seg(data_dir, electrode_list_path, electrode_name):
    """
    Load EEG signals from the Iowa dataset and extract the specified electrode.
    Signals are segmented into 2-second windows (500 Hz sampling rate) and flattened.
    """
    with open(electrode_list_path, "r", encoding="utf-8") as f:
        electrode_list = f.read().strip().split()
    
    if electrode_name not in electrode_list:
        raise ValueError(f"⚠️ The {electrode_name} electrode is not in the file electrode_list.txt.")
    
    electrode_index = electrode_list.index(electrode_name)
    print(f"✅ {electrode_name} found at index: {electrode_index}")
    
    with h5py.File(data_dir, 'r') as f:
        EEG_data = f['EEG']
        num_groups = 2  # PD and control
        num_patients = 14
        fs = 500  # Sampling frequency for Iowa dataset
        
        segmented_data = []
        labels = []
        
        ch_ref = EEG_data[electrode_index][0, 0] if EEG_data[electrode_index].shape == (1, 1) else EEG_data[electrode_index][0]
        if isinstance(ch_ref, h5py.Reference):
            ch_data = f[ch_ref]
        else:
            raise TypeError(f"❌ Error: ch_ref is not a HDF5 reference but {type(ch_ref)}")
        
        for group_idx in range(num_groups):
            group_ref = ch_data[group_idx][0] if ch_data[group_idx].shape == (1,) else ch_data[group_idx]
            if isinstance(group_ref, h5py.Reference):
                group_data = f[group_ref]
            else:
                raise TypeError(f"❌ Error: group_ref is not a HDF5 reference but {type(group_ref)}")
            
            for patient_idx in range(num_patients):
                patient_ref = group_data[patient_idx][0] if group_data[patient_idx].shape == (1,) else group_data[patient_idx]
                if isinstance(patient_ref, h5py.Reference):
                    signal = f[patient_ref][:].squeeze()
                    segments = segment_signal(signal, fs)
                    segmented_data.extend(segments)  # Flatten list of segments
                    labels.extend([group_idx] * len(segments))  # Extend labels accordingly
                else:
                    raise TypeError(f"❌ Error: patient_ref is not a HDF5 reference but {type(patient_ref)}")
    
    print(f"✅ {len(segmented_data)} total segments loaded.")
    return segmented_data, labels

def load_data_sandiego_1D_seg(data_dir, electrode_list_path, electrode_name):
    """
    Loads EEG signals from the San Diego dataset, extracts specified electrode,
    and segments signals into 2-second windows (512 Hz sampling rate), flattening the output.
    """
    with open(electrode_list_path, "r", encoding="utf-8") as f:
        electrode_list = f.read().strip().split()
    
    if electrode_name not in electrode_list:
        raise ValueError(f"⚠️ The {electrode_name} electrode is not in the electrode list.")
    
    electrode_index = electrode_list.index(electrode_name)
    print(f"✅ {electrode_name} found at index: {electrode_index}")
    
    data = []
    labels = []
    fs = 512  # Sampling frequency for San Diego dataset
    mne.set_log_level("ERROR")
    
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
                        segments = load_bdf_1D(file_path, electrode_index)
                        data.extend(segments)
                        labels.extend([label] * len(segments))
        
        elif sub_folder.startswith("sub-pd"):  # Parkinson's disease group
            for session, label in [("ses-off", 1), ("ses-on", 2)]:
                session_path = os.path.join(sub_path, session, "eeg")
                if os.path.exists(session_path):
                    for file in os.listdir(session_path):
                        if file.endswith(".bdf"):
                            file_path = os.path.join(session_path, file)
                            segments = load_bdf_1D(file_path, electrode_index)
                            data.extend(segments)
                            labels.extend([label] * len(segments))
    
    print(f"✅ {len(data)} total segments loaded.")
    return data, labels
