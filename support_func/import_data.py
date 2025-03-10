import numpy as np
import h5py

def load_data_iowa(data_dir):
    with h5py.File(data_dir, 'r') as f:
        EEG_data = f['EEG']

        num_channels = 63
        num_groups = 2  # pd, control
        num_patients = 14

        table = []

        for ch in range(num_channels):
            # Déréférencement du channel
            ch_ref = EEG_data[ch][0, 0] if EEG_data[ch].shape == (1, 1) else EEG_data[ch][0]
            ch_data = f[ch_ref]

            channel = []
            for group_idx in range(num_groups):
                group_ref = ch_data[group_idx][0] if ch_data[group_idx].shape == (1,) else ch_data[group_idx]
                group_data = f[group_ref]

                group = []
                for patient_idx in range(num_patients):
                    patient_ref = group_data[patient_idx][0] if group_data[patient_idx].shape == (1,) else group_data[patient_idx]
                    signal = f[patient_ref][:]
                    signal = signal.squeeze()
                    group.append(signal)

                channel.append(group)
            table.append(channel)
    table_t = transpose_and_flatten(table)  
    print(len(table_t),len(table_t[0]),len(table_t[0][0]))
    return table_t 

def transpose_and_flatten(table):
    num_channels = len(table)
    num_groups = len(table[0])
    num_patients = len(table[0][0])

    # Étape 1 : Transpose -> table_transposed[group][patient][channel]
    table_transposed = []
    for group_idx in range(num_groups):
        group = []
        for patient_idx in range(num_patients):
            patient = []
            for ch_idx in range(num_channels):
                signal = table[ch_idx][group_idx][patient_idx]
                patient.append(signal)
            group.append(patient)
        table_transposed.append(group)

    # Étape 2 : Flattening -> table_flattened[patient][channel]
    table_flattened = []

    for group in table_transposed:
        for patient in group:
            table_flattened.append(patient)

    return table_flattened

def load_labels_iowa():
    sub_list = np.zeros(28)
    for i in range(28):
        if i<=13:
            sub_list[i] = 1
        
    return sub_list


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

    cpz_index = electrode_list.index(electrode_name)  # Find CPz index
    print(f"✅ {electrode_name} found at the index : {cpz_index}")

    with h5py.File(data_dir, 'r') as f:
        EEG_data = f['EEG']
        print("📂 Keys available in the HDF5 file :", list(f.keys()))

        num_groups = 2  # PD and control
        num_patients = 14

        cpz_data = []  # List to story only CPz's eegs


        ch_ref = EEG_data[cpz_index][0, 0] if EEG_data[cpz_index].shape == (1, 1) else EEG_data[cpz_index][0]
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
                    cpz_data.append(signal)
                else:
                    raise TypeError(f"❌ Erreur : patient_ref is not a HDF5 reference but {type(patient_ref)}")

    print(f"✅ Extraction finished, {len(cpz_data)} signals loaded.")
    return cpz_data  # Liste contenant 28 signaux (T,)

def load_labels_iowa_1D():
    """
    Set up the labels for the 28 patients.
    """
    labels = np.zeros(28)
    labels[:14] = 1  # 14 first patients = Parkinson
    return labels


#lab = load_data_iowa_1D("./Data/iowa/IowaData.mat",'./Data/iowa/electrode_list.txt')
