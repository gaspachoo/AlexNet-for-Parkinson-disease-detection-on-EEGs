import torch
import matplotlib.pyplot as plt
import numpy as np

dataset1 = torch.load("train_sd_off.pt")
dataset2 = torch.load("val_sd_off.pt")

def plot_wst_examples_colormap(dataset, title, num_samples=4, num_rows= 2):
    images = dataset['tfr']
    labels = dataset['labels']
    
    # Compute global vmin and vmax for consistent color scaling
    vmin, vmax = np.min([img.numpy() for img in images]), np.max([img.numpy() for img in images])
    
    # Select indices for each class (HC: 0, PD: 1)
    hc_indices = [i for i, label in enumerate(labels) if label == 0][:num_samples]
    
    if num_rows == 2:
        pd_indices = [i for i, label in enumerate(labels) if label == 1][:num_samples]     
    else:
        pdoff_indices = [i for i, label in enumerate(labels) if label == 1][:num_samples]
        pdon_indices = [i for i, label in enumerate(labels) if label == 2][:num_samples]
         
    # Create subplots
    fig, axes = plt.subplots(num_rows, num_samples, figsize=(5 * num_samples, 8), sharex=True, sharey=True)
    
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

    if num_rows==2:
        # Plot PD samples (second row)
        for idx, sample_idx in enumerate(pd_indices):
            ax = axes[1, idx]
            img = images[sample_idx].numpy()

            im = ax.imshow(img, cmap='viridis', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
            ax.set_title(f"PD Sample {sample_idx}")

            # Remove redundant labels
            if idx == 0:
                ax.set_ylabel("Frequency")
            else:
                ax.set_yticklabels([])
            
    else:        
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
    for row in range(num_rows-1):
        for ax in axes[row]:
            ax.set_xticklabels([])
    for ax in axes[-1]:
        ax.set_xlabel("Time")

    # Add a single colorbar to the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust position for clarity
    fig.colorbar(im, cax=cbar_ax)

    # Add a global title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit colorbar
    plt.show()

# Plot examples for both datasets
plot_wst_examples_colormap(dataset1, "TFR via WST - Dataset 1 (Train)")
plot_wst_examples_colormap(dataset2, "TFR via WST - Dataset 2 (Validation)")
