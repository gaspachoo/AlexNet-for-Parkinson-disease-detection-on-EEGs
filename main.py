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
    model,
    num_epochs=500,
    batch_size=20,
    learning_rate=1e-4,
    patience=8,
    checkpoint_path="./Checkpoints/checkpoint.pth",  # Path for checkpoint
):
    """
    Train and validate a model on precomputed EEG data with early stopping, checkpoint saving, and metrics.

    Returns:
        model (nn.Module): Trained model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # --- Reproducibility: set global seeds and deterministic flags ---
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Force deterministic algorithms where possible (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Use a seeded Generator for DataLoader shuffling
    dl_generator = torch.Generator()
    dl_generator.manual_seed(seed)

    print("Creating DataLoaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        generator=dl_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        generator=dl_generator,  # included for determinism
    )

    # Model, loss, optimizer, metrics
    if model.lower() == "AlexNet".lower():
        model = AlexNetCustom(num_classes=2).to(device)
    elif model.lower() == "ResNet".lower():
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    else:
        raise ValueError("Model name not implemented yet")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    f1_metric = MulticlassF1Score(num_classes=2, average="macro").to(device)
    confmat_metric = MulticlassConfusionMatrix(num_classes=2).to(device)

    # Early stopping and checkpoint setup
    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 0
    best_model_state = model.state_dict()

    # Resume from checkpoint if it exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_loss = checkpoint["best_val_loss"]
        patience_counter = checkpoint["patience_counter"]
        start_epoch = checkpoint["epoch"] + 1
        best_model_state = checkpoint["best_model_state"]
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}...")

    wandb.init(
        project="A2 PD Detection",
        name=f"{model}_{os.path.basename(checkpoint_path)}",
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model": model,
        },
    )

    print("Training")
    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for batch in train_loader:
                images, labels = batch
                images = images.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True).long()

                optimizer.zero_grad()
                outputs = model(images)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

            train_loss = running_loss / len(train_loader)
            train_accuracy = 100.0 * correct_train / total_train

            # Validation
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            f1_metric.reset()
            confmat_metric.reset()

            with torch.no_grad():
                all_preds = []
                all_labels = []

                for batch in val_loader:
                    images, labels = batch
                    images = images.to(device, non_blocking=True).float()
                    labels = labels.to(device, non_blocking=True).long()

                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item()

                    _, predicted = torch.max(outputs, 1)
                    correct_val += (predicted == labels).sum().item()
                    total_val += labels.size(0)

                    f1_metric.update(predicted, labels)
                    confmat_metric.update(predicted, labels)

                    # Ajout pour wandb
                    all_preds.append(predicted.cpu())
                    all_labels.append(labels.cpu())

                all_preds = torch.cat(all_preds)
                all_labels = torch.cat(all_labels)

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100.0 * correct_val / total_val
            val_f1 = f1_metric.compute()

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Val F1: {val_f1:.4f}"
            )

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

            # Early Stopping + Save checkpoint if improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict()

                # Save checkpoint
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
                print("Checkpoint saved.")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        print("Training complete.")
        model.load_state_dict(best_model_state)

        # Final confusion matrix
        cm = confmat_metric.compute().cpu().numpy()
        cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=["Control", "PD"],
            yticklabels=["Control", "PD"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix on Validation Set")
        plt.show()

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


if __name__ == "__main__":
    file_end = "sd_off_Fz"
    model_name = "resnet"
    train_dataset = torch.load(f"./Datasets_pt/train_{file_end}.pt")
    val_dataset = torch.load(f"./Datasets_pt/val_{file_end}.pt")

    print("Dataset loaded...")

    trained_model = train_and_validate(
        train_dataset,
        val_dataset,
        model_name,
        num_epochs=200,
        patience=15,
        checkpoint_path=f"./Checkpoints/checkpoint_{model_name}_{file_end}.pth",  # Custom checkpoint path
    )

    torch.save(
        trained_model.state_dict(),
        f"./Checkpoints/model_{model_name}_{file_end}.pth",
    )
