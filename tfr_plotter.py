import torch
import matplotlib.pyplot as plt
import numpy as np

# ✅ Load dataset (list of tuples)
dataset_train = torch.load("./Datasets_pt/train_iowa.pt")
dataset_val = torch.load("./Datasets_pt/val_iowa.pt")


def plot_examples(dataset, title, num_samples=11):
    # ✅ Unzip dataset into images and labels
    images, labels = zip(*dataset)  
    images = torch.stack(images)  # Convert list to tensor (num_samples, 3, 227, 227)
    labels = torch.tensor(labels)  # Convert list to tensor
    
    unique_labels = sorted(set(labels.numpy()))  # Ensure labels are sorted for indexing
    num_rows = len(unique_labels)

    # ✅ Select indices for each class
    hc_indices = [i for i, label in enumerate(labels) if label == 0][:num_samples]
    
    if num_rows == 2:
        pd_indices = [i for i, label in enumerate(labels) if (label == 1 or label == 2)][:num_samples]     
    else:
        pdoff_indices = [i for i, label in enumerate(labels) if label == 1][:num_samples]
        pdon_indices = [i for i, label in enumerate(labels) if label == 2][:num_samples]

    # ✅ Create subplots
    fig, axes = plt.subplots(num_rows, num_samples, figsize=(4 * num_samples, 3 * num_rows), sharex=True, sharey=True)
    axes = np.array(axes)  # Ensure axes is a 2D NumPy array for indexing

    # ✅ Function to plot RGB images
    def plot_rgb(ax, img, title):
        img = img.permute(1, 2, 0).numpy()  # Convert from (3, H, W) to (H, W, 3)
        im = ax.imshow(img)  # RGB Image
        ax.set_title(title)
        ax.axis("on")
        return im

    # ✅ Plot HC samples (first row)
    for idx, sample_idx in enumerate(hc_indices):
        im = plot_rgb(axes[0, idx], images[sample_idx], f"HC {sample_idx}")
      
    if num_rows == 2:
        # ✅ Plot PD samples (second row)
        for idx, sample_idx in enumerate(pd_indices):
            med = 'On' if sorted(set(labels.numpy()))[1] == 2 else 'Off'
            im = plot_rgb(axes[1, idx], images[sample_idx], f"PD {med} {sample_idx}")


    else:
        # ✅ Plot PD OFF samples (second row)
        for idx, sample_idx in enumerate(pdoff_indices):
            im = plot_rgb(axes[1, idx], images[sample_idx], f"PD OFF {sample_idx}")


        # ✅ Plot PD ON samples (third row)
        for idx, sample_idx in enumerate(pdon_indices):
            im = plot_rgb(axes[2, idx], images[sample_idx], f"PD ON {sample_idx}")
            
    fig.text(0.001, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=12)
    fig.text(0.5, 0.001, 'Time', ha='center', fontsize=12)

        
    # Add a single colorbar to the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust position for clarity
    fig.colorbar(im, cax=cbar_ax)
    
    # ✅ Adjust layout
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  
    plt.show()

# ✅ Plot examples for both datasets
plot_examples(dataset_train, "TFR via WST - Dataset 1 (Train)",5)
plot_examples(dataset_val, "TFR via WST - Dataset 2 (Validation)",3)
