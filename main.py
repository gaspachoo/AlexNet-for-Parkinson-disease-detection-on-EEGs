from support_func.NN_classes import *
from dataset_preloader import *
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassF1Score, MulticlassConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns


def train_and_validate(
    train_dataset,
    val_dataset,
    num_epochs=500,
    batch_size=20,
    learning_rate=1e-4,
    patience=8
):
    """
    Train and validate a model on precomputed EEG data with early stopping and metrics.

    Returns:
        model (nn.Module): Trained model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    
    print("Creating DataLoaders...")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    print("Training")
    # Model, loss, optimizer, metrics
    #model = AlexNetCustom(num_classes=2).to(device)
    
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 classes (HC vs. PD)
    
    #model.load_state_dict(torch.load("./Models/model_resnet_sdoff-70p.pth"))
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # ✅ Adam optimizer with L2 Regularization (Weight Decay)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # L2 reg

    f1_metric = MulticlassF1Score(num_classes=2, average='macro').to(device)
    confmat_metric = MulticlassConfusionMatrix(num_classes=2).to(device)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            images, labels = batch  # ✅ Unpack tuple directly
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
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        f1_metric.reset()
        confmat_metric.reset()

        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch  # ✅ Unpack tuple directly
                images = images.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True).long()

                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

                f1_metric.update(predicted, labels)
                confmat_metric.update(predicted, labels)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100.0 * correct_val / total_val
        val_f1 = f1_metric.compute()

        print(f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Val F1: {val_f1:.4f}")

        # 🛑 Early Stopping
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
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_normalized, annot=True,  fmt=".2f", cmap='Blues',
        xticklabels=["Control", "PD"], yticklabels=["Control", "PD"]
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix on Validation Set')
    plt.show()

    return model

if __name__ == "__main__":
    file_end = "sd_off_Fz"
    train_dataset = torch.load(f"./Datasets_pt/train_{file_end}.pt")
    val_dataset = torch.load(f"./Datasets_pt/val_{file_end}.pt")
    
    print("Dataset loaded...")
    
    trained_model = train_and_validate(train_dataset, val_dataset, num_epochs=30, patience=10)
    torch.save(trained_model.state_dict(), f"./Models/model_resnet_{file_end}.pth")
