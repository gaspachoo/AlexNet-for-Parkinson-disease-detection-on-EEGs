"""
Dataset preprocessing script for EEG data.

This script loads raw EEG data, applies filtering and Wavelet Scattering Transform (WST),
then saves preprocessed datasets as .pt files for training.
"""

import argparse

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import support_func.wavelet_transform as wt
from support_func.dataset_class import EEGDataset_1D
from support_func.filters import SavGol_Wavelet, bandpass_filter


def process_and_save(indices, dataset, transform, fs):
    images_list = []
    labels_list = []
    n_indices = len(indices)

    for i, idx in enumerate(indices):
        sample = dataset[idx]  # e.g. {"eeg": shape (60600,), "label": 0/1}
        image = sample["eeg"]
        image_filtered = bandpass_filter(
            image.T, lowcut=0.5, highcut=40.0, fs=fs, order=4
        )
        image_filtered = SavGol_Wavelet(image_filtered).T
        # image_filtered = savgol_filter(image.T, window_length=11, polyorder=3).T
        sample["eeg"] = image_filtered
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
    mode, electrode_name, segment_duration, save=False, medication=None
):
    if mode == "iowa":
        folder_path = "./Data/iowa/IowaData.mat"
        electrode_list_path = "./Data/iowa/electrode_list.txt"
    elif mode == "san_diego":
        folder_path = "./Data/san_diego"
        electrode_list_path = "./Data/san_diego/electrode_list.txt"
    else:
        raise ValueError("Expected 'iowa' or 'san_diego'")

    raw_dataset = EEGDataset_1D(
        folder_path, electrode_name, electrode_list_path, segment_duration, medication
    )

    print("Raw dataset loaded, splitting.")

    labels = np.array([raw_dataset[i]["label"] for i in range(len(raw_dataset))])
    indices = np.arange(len(raw_dataset))

    train_idx, val_idx = train_test_split(
        indices, stratify=labels, test_size=0.2, random_state=42
    )

    print("Applying WST")
    fs = 512 if mode == "san_diego" else 500
    transform_wst = wt.WaveletScatteringTransform(
        segment_duration, fs, J=10, Q=24
    )  # Change WST hyperparameters

    print("Processing and saving")
    train_images, train_labels = process_and_save(
        train_idx, raw_dataset, transform_wst, fs
    )
    val_images, val_labels = process_and_save(val_idx, raw_dataset, transform_wst, fs)

    train_dataset = list(zip(train_images, train_labels))
    val_dataset = list(zip(val_images, val_labels))

    if save:
        path = "./Datasets_pt/"
        dataset_name = "iowa" if mode == "iowa" else f"sd_{medication or 'onandoff'}"
        train_path = path + f"train_{dataset_name}_{electrode_name}.pt"
        val_path = path + f"val_{dataset_name}_{electrode_name}.pt"

        torch.save(train_dataset, train_path)
        print(
            f"Saved {train_path} with shape {train_images.shape} and labels shape {train_labels.shape}"
        )
        torch.save(val_dataset, val_path)
        print(
            f"Saved {val_path} with shape {val_images.shape} and labels shape {val_labels.shape}"
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

    # Parse arguments
    args = parser.parse_args()

    # Display configuration
    print("\n" + "=" * 60)
    print("Dataset Preprocessing Configuration")
    print("=" * 60)
    print(f"Mode:              {args.mode}")
    print(f"Electrode:         {args.electrode}")
    print(f"Segment duration:  {args.segment_duration} seconds")
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
    )

    print("\nDataset preprocessing complete!")


if __name__ == "__main__":
    main()
