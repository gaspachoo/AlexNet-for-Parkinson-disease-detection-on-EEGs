"""
EEG Signal Filtering Comparison Tool

This script loads raw EEG data and compares various filtering techniques
using matplotlib subplots for visualization.
"""

import argparse
import os
import sys

import h5py
import matplotlib.pyplot as plt
import mne
import numpy as np

from support_func.filters import (
    SKLFast_ICA,
    bandpass_filter,
    matlab_like_cleaning,
    modern_cleaning,
    wavelet_denoising,
)


def load_iowa_all_channels(data_dir, electrode_list_path, subject_idx=0, group_idx=0):
    """
    Load ALL EEG channels from Iowa dataset for a specific subject.

    Parameters:
        data_dir: Path to IowaData.mat
        electrode_list_path: Path to electrode_list.txt
        subject_idx: Subject index within group (0-13)
        group_idx: Group index (0=PD, 1=Control)

    Returns:
        all_signals: 2D numpy array (n_channels, n_samples)
        electrode_list: List of electrode names
        fs: Sampling frequency
    """
    with open(electrode_list_path, "r", encoding="utf-8") as f:
        electrode_list = f.read().strip().split()

    fs = 500  # Iowa sampling frequency
    all_signals = []

    with h5py.File(data_dir, "r") as f:
        EEG_data = f["EEG"]

        for electrode_index in range(len(electrode_list)):
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
                group_data[subject_idx][0]
                if group_data[subject_idx].shape == (1,)
                else group_data[subject_idx]
            )
            patient_data = f[patient_ref]

            signal = patient_data[:].flatten()
            all_signals.append(signal)

    all_signals = np.array(all_signals)
    return all_signals, electrode_list, fs


def load_iowa_signal(
    data_dir, electrode_list_path, electrode_name, subject_idx=0, group_idx=0
):
    """
    Load a single EEG signal from Iowa dataset for a specific subject and electrode.

    Parameters:
        data_dir: Path to IowaData.mat
        electrode_list_path: Path to electrode_list.txt
        electrode_name: Name of electrode (e.g., 'AFz')
        subject_idx: Subject index within group (0-13)
        group_idx: Group index (0=PD, 1=Control)

    Returns:
        signal: 1D numpy array
        fs: Sampling frequency
    """
    with open(electrode_list_path, "r", encoding="utf-8") as f:
        electrode_list = f.read().strip().split()

    if electrode_name not in electrode_list:
        raise ValueError(f"Electrode {electrode_name} not found in electrode list.")

    electrode_index = electrode_list.index(electrode_name)
    fs = 500  # Iowa sampling frequency

    with h5py.File(data_dir, "r") as f:
        EEG_data = f["EEG"]

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
            group_data[subject_idx][0]
            if group_data[subject_idx].shape == (1,)
            else group_data[subject_idx]
        )
        patient_data = f[patient_ref]

        signal = patient_data[:].flatten()

    return signal, fs


def load_san_diego_all_channels(data_dir, subject_id, session="hc"):
    """
    Load ALL EEG channels from San Diego dataset for a specific subject.

    Parameters:
        data_dir: Path to san_diego directory
        subject_id: Subject identifier (e.g., 'hc1', 'pd11')
        session: Session type ('hc', 'on', 'off')

    Returns:
        all_signals: 2D numpy array (n_channels, n_samples)
        electrode_names: List of electrode names
        fs: Sampling frequency
    """
    # Construct file path
    if subject_id.startswith("hc"):
        bdf_path = os.path.join(
            data_dir,
            f"sub-{subject_id}",
            "ses-hc",
            "eeg",
            f"sub-{subject_id}_ses-hc_task-rest_eeg.bdf",
        )
    else:  # PD patient
        bdf_path = os.path.join(
            data_dir,
            f"sub-{subject_id}",
            f"ses-{session}",
            "eeg",
            f"sub-{subject_id}_ses-{session}_task-rest_eeg.bdf",
        )

    if not os.path.exists(bdf_path):
        raise FileNotFoundError(f"File not found: {bdf_path}")

    # Load with MNE
    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
    fs = raw.info["sfreq"]

    all_signals = raw.get_data()
    electrode_names = raw.ch_names

    return all_signals, electrode_names, fs


def load_san_diego_signal(data_dir, electrode_name, subject_id, session="hc"):
    """
    Load a single EEG signal from San Diego dataset for a specific subject and electrode.

    Parameters:
        data_dir: Path to san_diego directory
        electrode_name: Name of electrode (e.g., 'Fz')
        subject_id: Subject identifier (e.g., 'hc1', 'pd11')
        session: Session type ('hc', 'on', 'off')

    Returns:
        signal: 1D numpy array
        fs: Sampling frequency
    """
    # Construct file path
    if subject_id.startswith("hc"):
        bdf_path = os.path.join(
            data_dir,
            f"sub-{subject_id}",
            "ses-hc",
            "eeg",
            f"sub-{subject_id}_ses-hc_task-rest_eeg.bdf",
        )
    else:  # PD patient
        bdf_path = os.path.join(
            data_dir,
            f"sub-{subject_id}",
            f"ses-{session}",
            "eeg",
            f"sub-{subject_id}_ses-{session}_task-rest_eeg.bdf",
        )

    if not os.path.exists(bdf_path):
        raise FileNotFoundError(f"File not found: {bdf_path}")

    # Load with MNE
    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
    fs = raw.info["sfreq"]

    # Get electrode index
    if electrode_name not in raw.ch_names:
        raise ValueError(
            f"Electrode {electrode_name} not found in channels: {raw.ch_names}"
        )

    electrode_index = raw.ch_names.index(electrode_name)
    signal = raw.get_data()[electrode_index, :]

    return signal, fs


def apply_filtering_techniques(signal, all_channels, target_channel_idx, fs):
    """
    Apply various filtering techniques to the signal.

    For mono-channel techniques: apply to single channel signal
    For multi-channel techniques (ICA): apply to all channels, then extract target channel

    Parameters:
        signal: 1D numpy array (single target channel)
        all_channels: 2D numpy array (n_channels, n_samples) - all EEG channels
        target_channel_idx: Index of the target channel in all_channels
        fs: Sampling frequency

    Returns:
        Dictionary of filtered signals (all 1D arrays for the target channel)
    """
    results = {}

    # Original signal
    results["Raw Signal"] = signal

    # === MONO-CHANNEL TECHNIQUES ===
    # Bandpass filter
    results["Bandpass Filter"] = bandpass_filter(signal, lowcut=0.5, highcut=40, fs=fs)

    # Wavelet denoising
    results["Wavelet Denoising"] = wavelet_denoising(signal, wavelet="db4", level=4)

    # MATLAB-like cleaning (works on 2D array)
    signal_2d = signal.reshape(1, -1) if signal.ndim == 1 else signal
    matlab_cleaned = matlab_like_cleaning(signal_2d, polyorder=5, window_length=127)
    results["MATLAB-like Cleaning"] = (
        matlab_cleaned[0] if matlab_cleaned.ndim == 2 else matlab_cleaned
    )

    # Bandpass + Wavelet combo
    bp_signal = bandpass_filter(signal, lowcut=0.5, highcut=40, fs=fs)
    results["Bandpass + Wavelet"] = wavelet_denoising(bp_signal, wavelet="db4", level=4)

    # === MULTI-CHANNEL TECHNIQUES ===
    # Modern cleaning with MNE ICA (applied to all channels)
    if all_channels is not None and all_channels.shape[0] > 1:
        try:
            mne_cleaned = modern_cleaning(all_channels, sfreq=fs)
            results["Modern Cleaning (MNE ICA)"] = mne_cleaned[target_channel_idx]
        except Exception as e:
            print(f"Warning: Modern cleaning (MNE ICA) failed: {e}")

    # SKL Fast ICA (applied to all channels)
    if all_channels is not None and all_channels.shape[0] > 1:
        try:
            skl_cleaned = SKLFast_ICA(all_channels, lda=2.5)
            results["SKL Fast ICA"] = skl_cleaned[target_channel_idx]
        except Exception as e:
            print(f"Warning: SKL Fast ICA failed: {e}")

    return results


def plot_comparison(results, fs, duration=10):
    """
    Plot comparison of different filtering techniques using subplots.

    Parameters:
        results: Dictionary of filtered signals
        fs: Sampling frequency
        duration: Duration to display in seconds
    """
    n_techniques = len(results)
    fig, axes = plt.subplots(n_techniques, 1, figsize=(14, 3 * n_techniques))

    if n_techniques == 1:
        axes = [axes]

    # Limit to specified duration
    n_samples = int(duration * fs)
    time_vector = np.arange(n_samples) / fs

    for ax, (technique_name, signal) in zip(axes, results.items()):
        signal_segment = signal[:n_samples]
        ax.plot(time_vector, signal_segment, linewidth=0.5)
        ax.set_title(technique_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (µV)")
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_val = np.mean(signal_segment)
        std_val = np.std(signal_segment)
        ax.text(
            0.98,
            0.95,
            f"μ={mean_val:.2f}, σ={std_val:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()


def main():
    """
    Main entry point with CLI argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Compare EEG filtering techniques on raw signals."
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["iowa", "san_diego"],
        help="Dataset to use: 'iowa' or 'san_diego'.",
    )

    # Electrode selection
    parser.add_argument(
        "--electrode",
        type=str,
        required=True,
        help="Electrode name (e.g., 'AFz' for Iowa, 'Fz' for San Diego).",
    )

    # Subject selection
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="Subject identifier. For Iowa: group index (0=PD, 1=Control) and subject index (0-13) separated by comma (e.g., '0,5'). For San Diego: subject ID (e.g., 'hc1', 'pd11').",
    )

    # Optional: session for San Diego
    parser.add_argument(
        "--session",
        type=str,
        default="hc",
        choices=["hc", "on", "off"],
        help="Session for San Diego dataset (default: 'hc'). Use 'on' or 'off' for PD patients.",
    )

    # Optional: display duration
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration to display in seconds (default: 10).",
    )

    args = parser.parse_args()

    # Load signal based on dataset
    print("\n" + "=" * 60)
    print("EEG Signal Filtering Comparison")
    print("=" * 60)
    print(f"Dataset:    {args.dataset}")
    print(f"Electrode:  {args.electrode}")
    print(f"Subject:    {args.subject}")
    if args.dataset == "san_diego":
        print(f"Session:    {args.session}")
    print(f"Duration:   {args.duration}s")
    print("=" * 60 + "\n")

    try:
        if args.dataset == "iowa":
            # Parse subject argument (group_idx,subject_idx)
            group_idx, subject_idx = map(int, args.subject.split(","))

            data_dir = "./Data/iowa/IowaData.mat"
            electrode_list_path = "./Data/iowa/electrode_list.txt"

            print(
                f"Loading Iowa data: Group {group_idx} ({'PD' if group_idx == 0 else 'Control'}), Subject {subject_idx}..."
            )

            # Load ALL channels
            all_channels, electrode_names, fs = load_iowa_all_channels(
                data_dir, electrode_list_path, subject_idx, group_idx
            )

            # Get target channel index and signal
            if args.electrode not in electrode_names:
                raise ValueError(
                    f"Electrode {args.electrode} not found in electrode list."
                )
            target_channel_idx = electrode_names.index(args.electrode)
            signal = all_channels[target_channel_idx]

        else:  # san_diego
            data_dir = "./Data/san_diego"

            print(
                f"Loading San Diego data: Subject {args.subject}, Session {args.session}..."
            )

            # Load ALL channels
            all_channels, electrode_names, fs = load_san_diego_all_channels(
                data_dir, args.subject, args.session
            )

            # Get target channel index and signal
            if args.electrode not in electrode_names:
                raise ValueError(
                    f"Electrode {args.electrode} not found in channels: {electrode_names}"
                )
            target_channel_idx = electrode_names.index(args.electrode)
            signal = all_channels[target_channel_idx]

        print(
            f"✅ All channels loaded: {all_channels.shape[0]} channels, {all_channels.shape[1]} samples, {fs} Hz"
        )
        print(f"   Target channel: {args.electrode} (index {target_channel_idx})")
        print(f"   Duration: {len(signal) / fs:.2f} seconds\n")

        # Apply filtering techniques
        print("Applying filtering techniques...")
        filtered_results = apply_filtering_techniques(
            signal, all_channels, target_channel_idx, fs
        )
        print(f"✅ {len(filtered_results)} techniques applied\n")

        # Plot comparison
        print("Generating comparison plot...")
        plot_comparison(filtered_results, fs, duration=args.duration)

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
