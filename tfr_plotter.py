import torch
import matplotlib.pyplot as plt
import numpy as np

dataset1 = torch.load("train_sd.pt")
dataset2 = torch.load("val_sd.pt")

def plot_wst_examples_colormap_2r(dataset, title, num_samples=4):
    images = dataset['tfr']
    labels = dataset['labels']
    
    # Compute global vmin and vmax for consistent color scaling
    vmin, vmax = np.min([img.numpy() for img in images]), np.max([img.numpy() for img in images])
    
    # Select indices for each class (HC: 0, PD: 1)
    hc_indices = [i for i, label in enumerate(labels) if label == 0][:num_samples]
    pd_indices = [i for i, label in enumerate(labels) if label == 1][:num_samples]
    
    
    # Create subplots
    fig, axes = plt.subplots(2, num_samples, figsize=(5 * num_samples, 8), sharex=True, sharey=True)
    
    # Ensure axes is a 2D NumPy array for consistent indexing
    axes = np.array(axes)

    # Plot HC samples (first row)
    for idx, sample_idx in enumerate(hc_indices):
        ax = axes[0, idx]
        img = images[sample_idx].numpy()

        im = ax.imshow(img, cmap='viridis', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f"HC Sample {sample_idx}")
        
        # Remove redundant labels
        if idx == 0:
            ax.set_ylabel("Frequency")
        else:
            ax.set_yticklabels([])

    # Plot PD off samples (second row)
    for idx, sample_idx in enumerate(pd_indices):
        ax = axes[1, idx]
        img = images[sample_idx].numpy()

        im = ax.imshow(img, cmap='viridis', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f"PD OFF Sample {sample_idx}")

        # Remove redundant labels
        if idx == 0:
            ax.set_ylabel("Frequency")
        else:
            ax.set_yticklabels([])
            
    # Plot PD on samples (third row)
    for idx, sample_idx in enumerate(pdon_indices):
        ax = axes[2, idx]
        img = images[sample_idx].numpy()

        im = ax.imshow(img, cmap='viridis', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f"PD ON Sample {sample_idx}")

        # Remove redundant labels
        if idx == 0:
            ax.set_ylabel("Frequency")
        else:
            ax.set_yticklabels([])

    # Hide x-axis labels for all but the last row
    for ax in axes[0]:
        ax.set_xticklabels([])
    for ax in axes[1]:
        ax.set_xlabel("Time")

    # Add a single colorbar to the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust position for clarity
    fig.colorbar(im, cax=cbar_ax)

    # Add a global title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit colorbar
    plt.show()

def plot_wst_examples_colormap_3r(dataset, title, num_samples=4):
    images = dataset['tfr']
    labels = dataset['labels']
    
    # Compute global vmin and vmax for consistent color scaling
    vmin, vmax = np.min([img.numpy() for img in images]), np.max([img.numpy() for img in images])
    
    # Select indices for each class (HC: 0, PD: 1)
    hc_indices = [i for i, label in enumerate(labels) if label == 0][:num_samples]
    pdoff_indices = [i for i, label in enumerate(labels) if label == 1][:num_samples]
    pdon_indices = [i for i, label in enumerate(labels) if label == 2][:num_samples]
    
    
    # Create subplots
    fig, axes = plt.subplots(3, num_samples, figsize=(5 * num_samples, 8), sharex=True, sharey=True)
    
    # Ensure axes is a 2D NumPy array for consistent indexing
    axes = np.array(axes)

    # Plot HC samples (first row)
    for idx, sample_idx in enumerate(hc_indices):
        ax = axes[0, idx]
        img = images[sample_idx].numpy()

        im = ax.imshow(img, cmap='viridis', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f"HC Sample {sample_idx}")
        
        # Remove redundant labels
        if idx == 0:
            ax.set_ylabel("Frequency")
        else:
            ax.set_yticklabels([])

    # Plot PD off samples (second row)
    for idx, sample_idx in enumerate(pdoff_indices):
        ax = axes[1, idx]
        img = images[sample_idx].numpy()

        im = ax.imshow(img, cmap='viridis', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f"PD OFF Sample {sample_idx}")

        # Remove redundant labels
        if idx == 0:
            ax.set_ylabel("Frequency")
        else:
            ax.set_yticklabels([])
            
    # Plot PD on samples (third row)
    for idx, sample_idx in enumerate(pdon_indices):
        ax = axes[2, idx]
        img = images[sample_idx].numpy()

        im = ax.imshow(img, cmap='viridis', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f"PD ON Sample {sample_idx}")

        # Remove redundant labels
        if idx == 0:
            ax.set_ylabel("Frequency")
        else:
            ax.set_yticklabels([])

    # Hide x-axis labels for all but the last row
    for ax in axes[0]:
        ax.set_xticklabels([])
    for ax in axes[1]:
        ax.set_xlabel("Time")

    # Add a single colorbar to the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust position for clarity
    fig.colorbar(im, cax=cbar_ax)

    # Add a global title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit colorbar
    plt.show()

# Plot examples for both datasets
plot_wst_examples_colormap_2r(dataset1, "TFR via WST - Dataset 1 (Train)")
plot_wst_examples_colormap_2r(dataset2, "TFR via WST - Dataset 2 (Validation)")
