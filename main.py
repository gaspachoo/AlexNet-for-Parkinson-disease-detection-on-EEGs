"""
Main entry point for training EEG-based Parkinson's Disease detection models.

This script handles command-line argument parsing, dataset path composition,
and invokes the training/validation pipeline.
"""

import argparse
import os
import sys

import torch

from train_and_validate import train_and_validate


def main():
    """
    Parse command-line arguments, compose dataset paths, verify file existence,
    and launch training/validation.
    """
    # Define available models
    available_models = ["alexnet", "resnet"]

    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="Train a deep learning model for Parkinson's Disease detection from EEG data."
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
        "--model",
        type=str,
        required=True,
        choices=available_models,
        help=f"Model architecture to use. Available: {', '.join(available_models)}.",
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
        default="on",
        choices=["on", "off"],
        help="Medication status for San Diego dataset (default: 'on'). Ignored for Iowa.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum number of training epochs (default: 200).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Batch size for DataLoader (default: 20).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for optimizer (default: 1e-4).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience in epochs (default: 15).",
    )

    # Parse arguments
    args = parser.parse_args()

    # Compose dataset filename based on mode and parameters
    if args.mode == "iowa":
        # Iowa dataset naming convention: iowa_{electrode}
        dataset_prefix = f"iowa_{args.electrode}"
    else:  # san_diego
        # San Diego dataset naming convention: sd_{medication}_{electrode}
        dataset_prefix = f"sd_{args.medication}_{args.electrode}"

    # Construct full dataset paths
    train_dataset_path = f"./Datasets_pt/train_{dataset_prefix}.pt"
    val_dataset_path = f"./Datasets_pt/val_{dataset_prefix}.pt"

    # Verify that dataset files exist
    if not os.path.exists(train_dataset_path):
        print(f"Error: Training dataset file not found: {train_dataset_path}\n")

        # Build the exact command needed to generate the missing dataset
        if args.mode == "iowa":
            preload_cmd = f"uv run python dataset_preloader.py --mode {args.mode} --electrode {args.electrode}"
        else:  # san_diego
            preload_cmd = f"uv run python dataset_preloader.py --mode {args.mode} --electrode {args.electrode} --medication {args.medication}"

        print("To generate the required dataset files, please run:")
        print(f"   {preload_cmd}\n")
        sys.exit(1)

    if not os.path.exists(val_dataset_path):
        print(f"Error: Validation dataset file not found: {val_dataset_path}\n")

        # Build the exact command needed to generate the missing dataset
        if args.mode == "iowa":
            preload_cmd = f"uv run python dataset_preloader.py --mode {args.mode} --electrode {args.electrode}"
        else:  # san_diego
            preload_cmd = f"uv run python dataset_preloader.py --mode {args.mode} --electrode {args.electrode} --medication {args.medication}"

        print("To generate the required dataset files, please run:")
        print(f"   {preload_cmd}\n")
        sys.exit(1)

    # Load datasets
    print(f"Loading training dataset from: {train_dataset_path}")
    train_dataset = torch.load(train_dataset_path)
    print(f"Loading validation dataset from: {val_dataset_path}")
    val_dataset = torch.load(val_dataset_path)
    print("Datasets loaded successfully.")

    # Compose checkpoint path
    checkpoint_path = f"./Checkpoints/checkpoint_{args.model}_{dataset_prefix}.pth"
    final_model_path = f"./Checkpoints/model_{args.model}_{dataset_prefix}.pth"

    # Launch training and validation
    print("\nStarting training with configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  Model: {args.model}")
    print(f"  Electrode: {args.electrode}")
    if args.mode == "san_diego":
        print(f"  Medication: {args.medication}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Patience: {args.patience}")
    print(f"  Checkpoint path: {checkpoint_path}")
    print(f"  Final model path: {final_model_path}\n")

    trained_model = train_and_validate(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
    )

    # Save final trained model
    print(f"Saving final model to: {final_model_path}")
    torch.save(trained_model.state_dict(), final_model_path)
    print("Training complete. Model saved successfully.")


if __name__ == "__main__":
    main()
