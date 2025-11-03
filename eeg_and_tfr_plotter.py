"""
EEG and TFR Plotter: Visualize filtered EEG signal and its Time-Frequency Representation.

This script loads raw multi-channel EEG data, applies ICA filtering (MNE or SKL),
extracts a target electrode, segments the signal, computes WST, and displays
the filtered time-domain signal alongside its TFR coefficients.

Usage:
    uv run python eeg_and_tfr_plotter.py --mode iowa --electrode AFz --filter mne_ica --segment_idx 0
    uv run python eeg_and_tfr_plotter.py --mode san_diego --electrode Fz --medication off --filter skl_ica --segment_idx 5
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from kymatio.torch import Scattering1D

from support_func.filters import MNE_ICA_Wavelet, SKLFast_ICA
from support_func.import_data import load_all_channels_iowa, load_all_channels_sandiego


def segment_signal(signal, segment_length=1536, overlap=0.5):
    """
    Segment a 1D signal into overlapping windows.

    Parameters:
        signal: np.ndarray (n_samples,)
        segment_length: int, length of each segment
        overlap: float, overlap ratio (0.0 to 1.0)

    Returns:
        segments: list of np.ndarray
    """
    step = int(segment_length * (1 - overlap))
    segments = []

    for start in range(0, len(signal) - segment_length + 1, step):
        segment = signal[start : start + segment_length]
        segments.append(segment)

    return segments


def compute_wst(signal, J=10, Q=24, T=1536):
    """
    Compute Wavelet Scattering Transform as 2D time-frequency representation.

    Parameters:
        signal: np.ndarray (T,)
        J: int, max scale
        Q: int, number of wavelets per octave
        T: int, signal length

    Returns:
        scattering_coeffs: np.ndarray (n_scales, n_time_frames) - 2D TFR
    """
    scattering = Scattering1D(J=J, Q=Q, shape=(T,), max_order=1)

    signal_tensor = torch.from_numpy(signal).float().unsqueeze(0)
    Sx = scattering(signal_tensor)  # Shape: (1, n_coeffs, n_time_frames)

    # Get first-order scattering coefficients only (frequency scales)
    meta = scattering.meta()
    order_array = meta["order"]
    idx_first_order = np.where(order_array == 1)[0]

    # Extract first-order: shape (1, n_scales, n_time_frames)
    S1 = Sx[:, idx_first_order, :]

    # Remove batch dimension: (n_scales, n_time_frames)
    S1_np = S1.squeeze(0).numpy()

    return S1_np


def plot_eeg_and_tfr(eeg_segment, tfr, title="EEG and TFR"):
    """
    Plot EEG time-domain signal and its TFR side-by-side.

    Parameters:
        eeg_segment: np.ndarray (n_samples,), time-domain EEG signal
        tfr: np.ndarray (n_scales, n_time), TFR coefficients as 2D image
        title: str, plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Time-domain EEG signal
    time = np.arange(len(eeg_segment))
    axes[0].plot(time, eeg_segment, linewidth=0.8, color="#2E86AB")
    axes[0].set_xlabel("Sample Index", fontsize=12)
    axes[0].set_ylabel("Amplitude", fontsize=12)
    axes[0].set_title(
        "Filtered EEG Signal (Time Domain)", fontsize=14, fontweight="bold"
    )
    axes[0].grid(alpha=0.3)

    # Plot 2: TFR as 2D image (Time-Frequency)
    im = axes[1].imshow(tfr, aspect="auto", origin="lower", cmap="viridis")
    axes[1].set_xlabel("Time", fontsize=12)
    axes[1].set_ylabel("Frequency Scale", fontsize=12)
    axes[1].set_title(
        "Time-Frequency Representation (WST)", fontsize=14, fontweight="bold"
    )

    # Add colorbar
    plt.colorbar(im, ax=axes[1], label="Scattering Coefficient")

    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot filtered EEG signal and its TFR (WST) computed from raw data."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["iowa", "san_diego"],
        help="Dataset to use (iowa or san_diego)",
    )
    parser.add_argument(
        "--electrode",
        type=str,
        required=True,
        help="Target electrode (e.g., AFz, Fz, Cz)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        required=True,
        choices=["mne_ica", "skl_ica"],
        help="ICA filtering method to apply (mne_ica or skl_ica)",
    )
    parser.add_argument(
        "--medication",
        type=str,
        default=None,
        choices=["on", "off"],
        help="Medication state for San Diego dataset (required if mode=san_diego)",
    )
    parser.add_argument(
        "--segment_idx",
        type=int,
        default=0,
        help="Index of the segment to visualize (default: 0 = first segment)",
    )
    parser.add_argument(
        "--recording_idx",
        type=int,
        default=0,
        help="Index of the recording to use (default: 0 = first recording)",
    )

    args = parser.parse_args()

    # === Step 1: Load multi-channel data ===
    print(f"\n{'=' * 60}")
    print(f"Loading {args.mode.upper()} dataset")
    print(f"{'=' * 60}")

    if args.mode == "iowa":
        data_path = os.path.join("Data", "iowa", "IowaData.mat")
        electrode_file = os.path.join("Data", "iowa", "electrode_list.txt")
        recordings, electrodes, sampling_rate = load_all_channels_iowa(
            data_path, electrode_file
        )
    else:  # san_diego
        if args.medication is None:
            raise ValueError("--medication is required for San Diego dataset (on/off)")
        data_path = os.path.join("Data", "san_diego")
        electrode_file = os.path.join("Data", "san_diego", "electrode_list.txt")
        recordings, electrodes, sampling_rate = load_all_channels_sandiego(
            data_path, electrode_file, medication=args.medication
        )

    print(f"Loaded {len(recordings)} recordings")

    if len(recordings) == 0:
        raise ValueError("No recordings loaded. Check your dataset path and format.")

    if args.recording_idx >= len(recordings):
        raise ValueError(
            f"Recording index {args.recording_idx} out of range (max: {len(recordings) - 1})"
        )

    # === Step 2: Select a recording ===
    recording = recordings[args.recording_idx]
    multi_channel_data = recording["data"]  # Shape: (n_channels, n_samples)
    label = recording["label"]
    subject_id = recording.get("subject_id", "unknown")

    print(f"\nSelected recording {args.recording_idx}:")
    print(f"  Subject: {subject_id}")
    print(f"  Label: {label}")
    print(f"  Shape: {multi_channel_data.shape}")

    # === Step 3: Apply ICA filtering ===
    print(f"\n{'=' * 60}")
    print(f"Applying {args.filter.upper()} filtering")
    print(f"{'=' * 60}")

    if args.filter == "mne_ica":
        filtered_data = MNE_ICA_Wavelet(multi_channel_data, sfreq=sampling_rate)
    elif args.filter == "skl_ica":
        filtered_data = SKLFast_ICA(multi_channel_data)
    else:
        raise ValueError(f"Unknown filter method: {args.filter}")

    # === Step 4: Extract target electrode ===
    if args.electrode not in electrodes:
        raise ValueError(
            f"Electrode {args.electrode} not found. Available: {electrodes[:10]}..."
        )

    electrode_idx = electrodes.index(args.electrode)
    target_channel = filtered_data[electrode_idx, :]  # Shape: (n_samples,)

    print(f"\nExtracted electrode {args.electrode} (index {electrode_idx})")
    print(
        f"Signal length: {len(target_channel)} samples ({len(target_channel) / sampling_rate:.2f} s)"
    )

    # === Step 5: Segment the signal ===
    print(f"\n{'=' * 60}")
    print("Segmenting signal")
    print(f"{'=' * 60}")
    segments = segment_signal(target_channel, segment_length=1536, overlap=0.5)
    print(f"Created {len(segments)} segments")

    if args.segment_idx >= len(segments):
        raise ValueError(
            f"Segment index {args.segment_idx} out of range (max: {len(segments) - 1})"
        )

    selected_segment = segments[args.segment_idx]

    # === Step 6: Compute WST ===
    print(f"\nComputing WST for segment {args.segment_idx}...")
    tfr = compute_wst(selected_segment, J=10, Q=24, T=1536)
    print(f"  Signal shape: {selected_segment.shape}")
    print(f"  TFR shape: {tfr.shape}")

    # === Step 7: Plot ===
    title = (
        f"{args.mode.upper()} | {args.electrode} | Filter: {args.filter.upper()} | "
        f"Subject: {subject_id} | Label: {label} | Segment {args.segment_idx}/{len(segments)}"
    )
    if args.mode == "san_diego":
        title += f" | Med: {args.medication}"

    plot_eeg_and_tfr(selected_segment, tfr, title=title)


if __name__ == "__main__":
    main()
