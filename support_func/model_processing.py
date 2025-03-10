import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from support_func.data_processing import load_dataset,load_labels,split_data

class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)  # EEG data
        self.labels = torch.tensor(labels, dtype=torch.long)  # Labels as long for classification

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_gen(data_folder,labels_file,num_classes, test_size=0.2, sampling_mode = None):
    """Generates the train and gives the number of the channels and the number of timepoints of the data

    Args:
        data_folder (string): name of the data folder
        labels_file (string): _description_
        num_classes (int): number of final classes
        test_size (float, optional): % of the data used for testing. Defaults to 0.2.
        sampling_mode (string, optional): undersampling of oversampling. Defaults to None.

    Returns:
       train_loader,test_loader, num_channels, num_timepoints
    """
    
    # Train gen
    data = load_dataset(data_folder)
    labels = load_labels(labels_file)
    num_channels, num_timepoints = data.shape[1], data.shape[2]
    
    labels = labels - 1

    ## Select how many classes do we want
    min_label, max_label = labels.min(), labels.max()
    #Create uniform bins
    bins = np.linspace(min_label, max_label, num_classes + 1)[1:-1]  # First and last excluded for np.digitize
    grouped_labels = np.digitize(labels, bins)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(data, grouped_labels, test_size=test_size, random_state=42)
    
    # Loader gen
    if sampling_mode == 'oversampling':
        X_train,y_train = random_oversample(X_train,y_train)
    elif sampling_mode == 'undersampling':
        X_train,y_train = random_undersample(X_train,y_train)
    else:
        pass
    
    # Datasets and DataLoaders
    train_dataset = EEGDataset(X_train, y_train)
    test_dataset = EEGDataset(X_test, y_test)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader,test_loader, num_channels, num_timepoints

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_data, batch_labels in loader:
        # ⬇️ Transfert GPU
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_labels).sum().item()
        total += batch_labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train_with_early_stopping(model, train_loader, val_loader, device, num_epochs=50, patience=7, lr=0.001, criterion=nn.CrossEntropyLoss()):
    """Train given model with given train loader and validation loader. Please specify the device to use.

    Args:
        model (class): class of a Model
        train_loader (torch.utils.Data.DataLoader): train loader, given by train_gen func
        val_loader (torch.utils.Data.DataLoader): validation loader, given by train_gen func
        num_epochs (int, optional): max number of epochs (can be less if early stopping). Defaults to 50.
        patience (int, optional): patience before early stopping. Defaults to 7.
        lr (float, optional): adam gamma coefficient. Defaults to 0.001.
        criterion (_type_, optional): loss function. Defaults to nn.CrossEntropyLoss().

    Returns:
        model, model_history (ndarray)
    """
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"Using device: {device}")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion,device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"🛑 Early stopping at epoch {epoch+1}.")
            break

    return model, history

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_data, batch_labels in loader:
            # ⬇️ Transfert GPU
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def random_oversample(data, labels):

    unique_classes, counts = np.unique(labels, return_counts=True)
    max_count = counts.max()

    new_data = []
    new_labels = []

    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        repeats = max_count // len(cls_indices)
        remainder = max_count % len(cls_indices)

        # Duplique les échantillons
        new_data.append(np.repeat(data[cls_indices], repeats, axis=0))
        new_labels.append(np.repeat(labels[cls_indices], repeats))

        if remainder > 0:
            chosen = np.random.choice(cls_indices, remainder, replace=False)
            new_data.append(data[chosen])
            new_labels.append(labels[chosen])

    data_balanced = np.vstack(new_data)
    labels_balanced = np.concatenate(new_labels)
    return data_balanced, labels_balanced

def random_undersample(data, labels):
    from collections import Counter

    unique_classes, counts = np.unique(labels, return_counts=True)
    min_count = counts.min()

    new_data = []
    new_labels = []

    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        selected = np.random.choice(cls_indices, min_count, replace=False)
        new_data.append(data[selected])
        new_labels.append(labels[selected])

    data_balanced = np.vstack(new_data)
    labels_balanced = np.concatenate(new_labels)
    return data_balanced, labels_balanced
