"""
Train and validate a deep learning model for Parkinson's Disease detection from EEG data.

This module contains the main training loop, validation, metrics computation,
checkpoint management, early stopping, and Weights & Biases logging.
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)

import wandb
from support_func.NN_classes import AlexNetCustom


def train_and_validate(
    train_dataset,
    val_dataset,
    model_name,
    num_epochs=500,
    batch_size=20,
    learning_rate=1e-4,
    patience=8,
    checkpoint_path="./Checkpoints/checkpoint.pth",
):
    """
    Train and validate a model on precomputed EEG data with early stopping,
    checkpoint saving, and metrics tracking.

    Parameters
    ----------
    train_dataset : list of tuples
        Training dataset containing (image_tensor, label) tuples.
    val_dataset : list of tuples
        Validation dataset containing (image_tensor, label) tuples.
    model_name : str
        Name of the model to use ('alexnet' or 'resnet').
    num_epochs : int, optional
        Maximum number of training epochs (default: 500).
    batch_size : int, optional
        Batch size for DataLoader (default: 20).
    learning_rate : float, optional
        Learning rate for Adam optimizer (default: 1e-4).
    patience : int, optional
        Number of epochs to wait for improvement before early stopping (default: 8).
    checkpoint_path : str, optional
        Path to save/load training checkpoints (default: './Checkpoints/checkpoint.pth').

    Returns
    -------
    model : nn.Module
        The trained model with best validation loss weights loaded.
    """

    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # --- Reproducibility: set global seeds and deterministic flags ---
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Force deterministic algorithms where possible (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Use a seeded Generator for DataLoader shuffling (reproducibility)
    dl_generator = torch.Generator()
    dl_generator.manual_seed(seed)

    print("Creating DataLoaders...")

    # Training DataLoader with shuffling enabled
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        generator=dl_generator,
    )

    # Validation DataLoader (no shuffling needed)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        generator=dl_generator,  # included for determinism
    )

    # --- Model initialization based on model_name ---
    if model_name.lower() == "alexnet":
        model = AlexNetCustom(num_classes=2).to(device)
    elif model_name.lower() == "resnet":
        # Load pretrained ResNet18 and modify final layer for binary classification
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model = model.to(device)
    else:
        raise ValueError(
            f"Model '{model_name}' not implemented. Choose 'alexnet' or 'resnet'."
        )

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Metrics for evaluation
    f1_metric = MulticlassF1Score(num_classes=2, average="macro").to(device)
    confmat_metric = MulticlassConfusionMatrix(num_classes=2).to(device)

    # Early stopping and checkpoint setup
    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 0
    best_model_state = model.state_dict()

    # Resume from checkpoint if it exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_loss = checkpoint["best_val_loss"]
        patience_counter = checkpoint["patience_counter"]
        start_epoch = checkpoint["epoch"] + 1
        best_model_state = checkpoint["best_model_state"]
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}...")

    # Initialize Weights & Biases for experiment tracking
    wandb.init(
        project="A2 PD Detection",
        name=f"{model_name}_{os.path.basename(checkpoint_path)}",
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model": model_name,
        },
    )

    print("Starting training...")
    try:
        for epoch in range(start_epoch, num_epochs):
            # --- Training phase ---
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for batch in train_loader:
                images, labels = batch
                images = images.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True).long()

                # Zero gradients, forward pass, compute loss, backward pass, update weights
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Accumulate training metrics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

            # Compute average training loss and accuracy
            train_loss = running_loss / len(train_loader)
            train_accuracy = 100.0 * correct_train / total_train

            # --- Validation phase ---
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            # Reset metrics for this epoch
            f1_metric.reset()
            confmat_metric.reset()

            with torch.no_grad():
                all_preds = []
                all_labels = []

                for batch in val_loader:
                    images, labels = batch
                    images = images.to(device, non_blocking=True).float()
                    labels = labels.to(device, non_blocking=True).long()

                    # Forward pass
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item()

                    # Compute predictions and accumulate accuracy
                    _, predicted = torch.max(outputs, 1)
                    correct_val += (predicted == labels).sum().item()
                    total_val += labels.size(0)

                    # Update metrics
                    f1_metric.update(predicted, labels)
                    confmat_metric.update(predicted, labels)

                    # Store predictions and labels for wandb confusion matrix
                    all_preds.append(predicted.cpu())
                    all_labels.append(labels.cpu())

                # Concatenate all predictions and labels
                all_preds = torch.cat(all_preds)
                all_labels = torch.cat(all_labels)

            # Compute average validation metrics
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100.0 * correct_val / total_val
            val_f1 = f1_metric.compute()

            # Print epoch summary
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, "
                f"Val F1: {val_f1:.4f}"
            )

            # Log metrics to Weights & Biases
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_accuracy,
                    "val_f1": (val_f1.item() if hasattr(val_f1, "item") else val_f1),
                },
                step=epoch,
            )

            # --- Early Stopping & Checkpoint Saving ---
            if avg_val_loss < best_val_loss:
                # Improved validation loss: save checkpoint
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict()

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "best_model_state": best_model_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "patience_counter": patience_counter,
                    },
                    checkpoint_path,
                )
                print("Checkpoint saved (validation loss improved).")
            else:
                # No improvement: increment patience counter
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        print("Training complete.")

        # Load best model weights
        model.load_state_dict(best_model_state)

        # --- Final confusion matrix visualization ---
        cm = confmat_metric.compute().cpu().numpy()
        # Row-normalize with safe division (rows that sum to 0 remain zeros)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.divide(
            cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0
        )

        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            xticklabels=["Control", "PD"],
            yticklabels=["Control", "PD"],
            cbar=True,
            linewidths=0,
            linecolor=None,
        )
        # Match visual style to confusion_normalizer.py
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Rotate y tick labels vertically; remove tick marks
        for lbl in ax.get_yticklabels():
            lbl.set_rotation(90)
            lbl.set_va("center")
        ax.tick_params(length=0)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix on Validation Set")
        plt.tight_layout()
        plt.show()

        # Log confusion matrix to wandb
        wandb.log(
            {
                "confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=all_labels.numpy(),
                    preds=all_preds.numpy(),
                    class_names=["Control", "PD"],
                )
            }
        )

        # Remove checkpoint file after successful training
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print("Checkpoint file removed.")

        wandb.finish()
        return model

    except KeyboardInterrupt:
        # Handle manual interruption (Ctrl+C)
        print("\nTraining interrupted by user. Saving current state...")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_model_state": best_model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved at epoch {epoch}. You can resume later.")

        wandb.finish()
        return model
