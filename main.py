from support_func.model_processing import PrecomputedEEGDataset
from support_func.NN_classes import AlexNetCustom, SimpleEEGCNN
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassF1Score, MulticlassConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns


def train_and_validate(
    num_epochs=15,
    batch_size=20,
    learning_rate=1e-4,
    patience=5
):
    """
    Train and validate AlexNetCustom on precomputed EEG data with early stopping and metrics.

    Returns:
        model (nn.Module): Trained model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Load precomputed datasets
    train_dataset = PrecomputedEEGDataset("train_sd_off_rgb.pt")
    val_dataset = PrecomputedEEGDataset("val_sd_off_rgb.pt")
    

    print("Dataset loaded, creating DataLoaders...")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    print("Training")
    # Model, loss, optimizer, metrics
    #model = AlexNetCustom(num_classes=2).to(device)
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 classes (HC vs. PD)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # ✅ Define optimizer with separate learning rates for weights and biases
    optimizer = optim.Adam([
        {"params": [param for name, param in model.named_parameters() if "bias" not in name], "lr": learning_rate},
        {"params": [param for name, param in model.named_parameters() if "bias" in name], "lr": learning_rate * 20},
    ], lr=learning_rate)

    f1_metric = MulticlassF1Score(num_classes=2, average='macro').to(device)
    confmat_metric = MulticlassConfusionMatrix(num_classes=2).to(device)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True).float()
            labels = batch["label"].to(device, non_blocking=True).long()

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        f1_metric.reset()
        confmat_metric.reset()

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True).float()
                labels = batch["label"].to(device, non_blocking=True).long()

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                f1_metric.update(predicted, labels)
                confmat_metric.update(predicted, labels)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100.0 * correct / total
        val_f1 = f1_metric.compute()

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.2f}%, "
              f"Val F1: {val_f1:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Training complete.")
    model.load_state_dict(best_model_state)

    # Final confusion matrix
    cm = confmat_metric.compute().cpu().numpy()
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=["Control", "PD"], yticklabels=["Control", "PD"]
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix on Validation Set')
    plt.show()

    return model


if __name__ == "__main__":
    trained_model = train_and_validate(num_epochs=15)
