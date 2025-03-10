import torch
import matplotlib.pyplot as plt


def plot_wst_examples(dataset, title, num_samples=2):
    images = dataset['images']
    labels = dataset['labels']

    hc_indices = [i for i, label in enumerate(labels) if label == 0][:num_samples]
    pd_indices = [i for i, label in enumerate(labels) if label == 1][:num_samples]

    fig, axes = plt.subplots(2, num_samples, figsize=(5 * num_samples, 8))

    vmax = images.max().item()

    for idx, sample_idx in enumerate(hc_indices):
        ax = axes[0, idx]
        img = images[sample_idx].numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        im = ax.imshow(img / vmax)  # Normalisation avec vmax pour garder l’échelle cohérente
        ax.set_title(f"HC Sample {sample_idx}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx, sample_idx in enumerate(pd_indices):
        ax = axes[1, idx]
        img = images[sample_idx].numpy().transpose(1, 2, 0)
        im = ax.imshow(img / vmax)
        ax.set_title(f"PD Sample {sample_idx}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_wst_examples_gray(dataset, title, num_samples=2):
    images = dataset['images']
    labels = dataset['labels']

    hc_indices = [i for i, label in enumerate(labels) if label == 0][:num_samples]
    pd_indices = [i for i, label in enumerate(labels) if label == 1][:num_samples]

    fig, axes = plt.subplots(2, num_samples, figsize=(5 * num_samples, 8))

    vmax = images.max().item()  # Max value for normalization

    for idx, sample_idx in enumerate(hc_indices):
        ax = axes[0, idx]
        img = images[sample_idx].squeeze(0).numpy()  # Remove channel dimension (C, H, W) -> (H, W)
        im = ax.imshow(img / vmax, cmap="gray")  # Use grayscale colormap
        ax.set_title(f"HC Sample {sample_idx}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx, sample_idx in enumerate(pd_indices):
        ax = axes[1, idx]
        img = images[sample_idx].squeeze(0).numpy()
        im = ax.imshow(img / vmax, cmap="gray")
        ax.set_title(f"PD Sample {sample_idx}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_wst_examples_rgb(dataset, title, num_samples=2):
    images = dataset['images']
    labels = dataset['labels']

    # Find healthy control (HC) and Parkinson's disease (PD) samples
    hc_indices = [i for i, label in enumerate(labels) if label == 0][:num_samples]
    pd_indices = [i for i, label in enumerate(labels) if label == 1][:num_samples]

    fig, axes = plt.subplots(2, num_samples, figsize=(5 * num_samples, 8))
    fig.subplots_adjust(right=0.85)  # Adjust for colorbars

    for idx, sample_idx in enumerate(hc_indices):
        ax = axes[0, idx]
        img = images[sample_idx]  # PyTorch tensor (C, H, W)
        
        # Convert to NumPy for visualization
        img = img.numpy()

        # Normalize only if necessary
        if img.min() < 0 or img.max() > 1:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Transpose for matplotlib (C, H, W) → (H, W, C)
        img = img.transpose(1, 2, 0)

        im = ax.imshow(img)  # Display RGB image
        ax.set_title(f"HC Sample {sample_idx}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")

        # Add colorbar
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx, sample_idx in enumerate(pd_indices):
        ax = axes[1, idx]
        img = images[sample_idx]

        img = img.numpy()
        
        if img.min() < 0 or img.max() > 1:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        img = img.transpose(1, 2, 0)

        im = ax.imshow(img)  # Display RGB image
        ax.set_title(f"PD Sample {sample_idx}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")

        # Add colorbar
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

dataset1 = torch.load("train_WST_tfr.pt")
dataset2 = torch.load("val_WST_tfr.pt")

def plot_wst_examples_grayscale(dataset, title, num_samples=4):
    images = dataset['tfr']
    labels = dataset['labels']

    hc_indices = [i for i, label in enumerate(labels) if label == 0][:num_samples]
    pd_indices = [i for i, label in enumerate(labels) if label == 1][:num_samples]

    fig, axes = plt.subplots(2, num_samples, figsize=(5 * num_samples, 8))

    for idx, sample_idx in enumerate(hc_indices):
        ax = axes[0, idx]
        img = images[sample_idx].numpy()  # ✅ Ensure it's 2D

        im = ax.imshow(img, cmap='viridis', aspect='auto', origin='lower') # ✅ Use viridis for colormap
        ax.set_title(f"HC Sample {sample_idx}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx, sample_idx in enumerate(pd_indices):
        ax = axes[1, idx]
        img = images[sample_idx].numpy()  

        im = ax.imshow(img, cmap='viridis', aspect='auto', origin='lower') 
        ax.set_title(f"PD Sample {sample_idx}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
# Plot examples for both datasets
plot_wst_examples_grayscale(dataset1, "TFR via WST - Dataset 1 (Train)")
plot_wst_examples_grayscale(dataset2, "TFR via WST - Dataset 2 (Validation)")

def plot_scattering_tfr(tfr_tensor, label_str=""):
    """
    tfr_tensor: shape [num_scales, T'] (torch Tensor)
    label_str: e.g. "HC" or "PD"
    """
    tfr_np = tfr_tensor.cpu().numpy()  # shape (freq, time)
    
    fig, ax = plt.subplots(figsize=(5,4))
    cax = ax.imshow(tfr_np, cmap='viridis', aspect='auto',
                    origin='lower',
                    extent=[0, tfr_np.shape[1], 0, tfr_np.shape[0]])
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency (Scales)")
    ax.set_title(label_str)
    fig.colorbar(cax, ax=ax)
    plt.show()