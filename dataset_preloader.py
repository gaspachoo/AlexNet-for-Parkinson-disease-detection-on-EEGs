"""
Dataset preprocessing script for EEG data.

This script loads raw EEG data, applies filtering and Wavelet Scattering Transform (WST),
then saves preprocessed datasets as .pt files for training.

Now supports multi-channel ICA methods by loading all channels, applying ICA,
then segmenting and extracting the target electrode.
"""

import argparse

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import support_func.wavelet_transform as wt
from support_func.dataset_class import EEGDataset_1D, EEGDataset_MultiChannel
from support_func.filters import (
    MNE_ICA_Wavelet,
    SavGol_Wavelet,
    SKLFast_ICA,
    bandpass_filter,
    wavelet_denoising,
)


def process_and_save(
    indices, dataset, transform, fs, filter_method="bandpass", use_multichannel=False
):
    """
    Process EEG samples with specified filtering method and WST.

    Parameters:
        indices: Sample indices to process
        dataset: Raw EEG dataset (EEGDataset_1D or EEGDataset_MultiChannel)
        transform: WST transform to apply
        fs: Sampling frequency
        filter_method: Filtering method to use
        use_multichannel: If True, filtering was already applied in dataset (ICA methods)

    Note: For multi-channel ICA methods, filtering is applied during dataset loading
          on continuous recordings before segmentation. For mono-channel methods,
          filtering is applied here to each segment.

    Returns:
        all_images: Stacked WST-transformed images
        all_labels: Tensor of labels
    """
    images_list = []
    labels_list = []
    n_indices = len(indices)

    for i, idx in enumerate(indices):
        sample = dataset[idx]  # e.g. {"eeg": shape (60600,), "label": 0/1}
        image = sample["eeg"]

        # Apply selected filtering method (only for mono-channel pipeline)
        if use_multichannel:
            # For ICA methods, filtering was already applied in dataset
            image_filtered = image.T
        elif filter_method == "none":
            image_filtered = image.T
        elif filter_method == "bandpass":
            image_filtered = bandpass_filter(
                image.T, lowcut=1.0, highcut=40.0, fs=fs, order=4
            )
        elif filter_method == "wavelet":
            bp_signal = bandpass_filter(
                image.T, lowcut=1.0, highcut=40.0, fs=fs, order=4
            )
            image_filtered = wavelet_denoising(bp_signal, wavelet="db4", level=4)
        elif filter_method == "savgol":
            bp_signal = bandpass_filter(
                image.T, lowcut=1.0, highcut=40.0, fs=fs, order=4
            )
            # SavGol_Wavelet expects 2D input (channels, samples)
            signal_2d = bp_signal.reshape(1, -1)
            image_filtered = SavGol_Wavelet(signal_2d, polyorder=5, window_length=127)
            image_filtered = (
                image_filtered[0] if image_filtered.ndim == 2 else image_filtered
            )
        else:
            raise ValueError(
                f"Unknown filter method: {filter_method}. "
                "Choose from: 'none', 'bandpass', 'wavelet', 'savgol'. "
                "Note: Multi-channel ICA methods are not yet supported for dataset preprocessing."
            )

        sample["eeg"] = image_filtered.T
        out = transform(sample)
        images_list.append(out["image"])
        labels_list.append(out["label"])

        av = 100 * (i + 1) / n_indices
        print(f"Progress : {av:.2f}%")

    # Stack along dimension 0 => shape [N, scales, time']
    all_images = torch.stack(images_list, dim=0)
    all_labels = torch.tensor(labels_list, dtype=torch.long)

    return all_images, all_labels


def preload_dataset(
    mode,
    electrode_name,
    segment_duration,
    save=False,
    medication=None,
    filter_method="bandpass",
):
    """
    Load and preprocess EEG dataset with specified filtering method.

    For ICA methods (mne_ica, skl_ica), uses EEGDataset_MultiChannel which:
    1. Loads all EEG channels for each subject
    2. Applies ICA filtering to continuous recordings
    3. Segments the filtered data
    4. Extracts the target electrode

    For mono-channel methods, uses traditional EEGDataset_1D.
    """
    if mode == "iowa":
        folder_path = "./Data/iowa/IowaData.mat"
        electrode_list_path = "./Data/iowa/electrode_list.txt"
    elif mode == "san_diego":
        folder_path = "./Data/san_diego"
        electrode_list_path = "./Data/san_diego/electrode_list.txt"
    else:
        raise ValueError("Expected 'iowa' or 'san_diego'")

    fs = 512 if mode == "san_diego" else 500

    # Determine if we need multi-channel dataset (for ICA methods)
    use_multichannel = filter_method in ["mne_ica", "skl_ica"]

    if use_multichannel:
        print(f"\n{'=' * 60}")
        print(f"Using MULTI-CHANNEL pipeline for {filter_method.upper()}")
        print(f"{'=' * 60}\n")

        # Define filter function based on method
        if filter_method == "mne_ica":
            filter_func = MNE_ICA_Wavelet
        elif filter_method == "skl_ica":
            filter_func = SKLFast_ICA
        else:
            filter_func = None

        # Use multi-channel dataset
        raw_dataset = EEGDataset_MultiChannel(
            folder_path,
            electrode_name,
            electrode_list_path,
            segment_duration,
            filter_func=filter_func,
            medication=medication,
        )
    else:
        print(f"\n{'=' * 60}")
        print(f"Using MONO-CHANNEL pipeline for {filter_method.upper()}")
        print(f"{'=' * 60}\n")

        # Use traditional single-channel dataset
        raw_dataset = EEGDataset_1D(
            folder_path,
            electrode_name,
            electrode_list_path,
            segment_duration,
            medication,
        )

    print("Dataset loaded, splitting into train/val...")

    labels = np.array([raw_dataset[i]["label"] for i in range(len(raw_dataset))])
    indices = np.arange(len(raw_dataset))

    train_idx, val_idx = train_test_split(
        indices, stratify=labels, test_size=0.2, random_state=42
    )

    print(f"Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")
    print("\nApplying WST transformation...")

    transform_wst = wt.WaveletScatteringTransform(segment_duration, fs, J=10, Q=24)

    print("Processing train set...")
    train_images, train_labels = process_and_save(
        train_idx, raw_dataset, transform_wst, fs, filter_method, use_multichannel
    )

    print("Processing validation set...")
    val_images, val_labels = process_and_save(
        val_idx, raw_dataset, transform_wst, fs, filter_method, use_multichannel
    )

    train_dataset = list(zip(train_images, train_labels))
    val_dataset = list(zip(val_images, val_labels))

    if save:
        path = "./Datasets_pt/"
        dataset_name = "iowa" if mode == "iowa" else f"sd_{medication or 'onandoff'}"
        # Add filter method to filename for clarity
        filter_suffix = f"_{filter_method}" if filter_method != "bandpass" else ""
        train_path = path + f"train_{dataset_name}_{electrode_name}{filter_suffix}.pt"
        val_path = path + f"val_{dataset_name}_{electrode_name}{filter_suffix}.pt"

        torch.save(train_dataset, train_path)
        print(
            f"\nSaved {train_path}\n  Shape: {train_images.shape}, Labels: {train_labels.shape}"
        )
        torch.save(val_dataset, val_path)
        print(
            f"Saved {val_path}\n  Shape: {val_images.shape}, Labels: {val_labels.shape}"
        )

    return train_dataset, val_dataset


def main():
    """
    Main entry point for dataset preprocessing with CLI arguments.
    """
    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="Preprocess EEG datasets and save as .pt files with WST transformation."
    )

    # Required arguments
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["iowa", "san_diego"],
        help="Dataset mode: 'iowa' or 'san_diego'.",
    )
    parser.add_argument(
        "--electrode",
        type=str,
        required=True,
        help="Electrode name (e.g., 'AFz' for Iowa, 'Fz' for San Diego).",
    )

    # Optional arguments
    parser.add_argument(
        "--medication",
        type=str,
        default=None,
        choices=["on", "off", None],
        help="Medication status for San Diego dataset (default: None). Use 'on' or 'off' for binary classification. Ignored for Iowa.",
    )
    parser.add_argument(
        "--segment_duration",
        type=int,
        default=5,
        help="Segment duration in seconds (default: 5).",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="bandpass",
        choices=["none", "bandpass", "wavelet", "savgol", "mne_ica", "skl_ica"],
        help="Filtering method to apply (default: 'bandpass'). "
        "Mono-channel: 'none', 'bandpass' (1-40 Hz), 'wavelet' (bandpass + wavelet denoising), 'savgol' (bandpass + SavGol-Wavelet). "
        "Multi-channel ICA: 'mne_ica' (MNE ICA + Wavelet), 'skl_ica' (sklearn FastICA).",
    )

    # Parse arguments
    args = parser.parse_args()

    # Display configuration
    print("\n" + "=" * 60)
    print("Dataset Preprocessing Configuration")
    print("=" * 60)
    print(f"Mode:              {args.mode}")
    print(f"Electrode:         {args.electrode}")
    print(f"Segment duration:  {args.segment_duration} seconds")
    print(f"Filter method:     {args.filter}")
    if args.mode == "san_diego":
        print(f"Medication:        {args.medication or 'all (on + off)'}")
    print("=" * 60 + "\n")

    # Run preprocessing
    train_set, validate_set = preload_dataset(
        mode=args.mode,
        electrode_name=args.electrode,
        segment_duration=args.segment_duration,
        save=True,
        medication=args.medication,
        filter_method=args.filter,
    )

    print("\nDataset preprocessing complete!")


if __name__ == "__main__":
    main()
