import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassF1Score, MulticlassConfusionMatrix
import torch
import seaborn as sns

def plot_history(history, model_name):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title(f"{model_name} Training History")
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    plt.show()

def evaluate_with_metrics_gpu(model, loader, device):
    model.eval()

    f1_metric = MulticlassF1Score(num_classes=3, average='weighted').to(device)
    cm_metric = MulticlassConfusionMatrix(num_classes=3).to(device)

    with torch.no_grad():
        for batch_data, batch_labels in loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_data)
            _, preds = torch.max(outputs, 1)

            f1_metric.update(preds, batch_labels)
            cm_metric.update(preds, batch_labels)

    f1_score_gpu = f1_metric.compute()
    confusion_matrix_gpu = cm_metric.compute()

    print(f"Weighted F1-score (GPU): {f1_score_gpu:.4f}")

    # Passage sur CPU et conversion en numpy
    cm_cpu = confusion_matrix_gpu.cpu().numpy()

    # 🔹 Normalisation par classe réelle (ligne)
    cm_normalized = cm_cpu.astype('float') / cm_cpu.sum(axis=1, keepdims=True)

    # Heatmap normalisée
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Low', 'Medium', 'High'],
                yticklabels=['Low', 'Medium', 'High'],
                cbar_kws={'label': 'Proportion'})
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.show()
