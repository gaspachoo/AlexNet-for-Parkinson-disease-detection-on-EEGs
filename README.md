# 🧠 PD Detection framework : Applying WST and AlexNet on EEGs

## 📌 Project Overview

This project focuses on implementing and evaluating different neural network architectures for EEG classification using **PyTorch**. It EEG datasets in order to be able to recognize if a patient is suffering from PD thanks to a simple EEG of a few minutes.

## 📂 Project Structure

The repository is organized as follows:

- **`dataset_preloader.py`** – CLI tool for EEG dataset preprocessing with Wavelet Scattering Transform (WST). Generates `.pt` files ready for training.
- **`main.py`** – Main CLI entry point for model training with argparse interface.
- **`train_and_validate.py`** – Core training and validation logic module, including DataLoader setup, training loop, metrics, early stopping, and checkpointing.
- **`compare_filtering.py`** – CLI tool to compare raw EEG signals with various filtering techniques using matplotlib subplots.
- **`tfr_plotter.py`** – Generates visualizations for EEG signal transformations.
- **`support_func/`** – Utility functions supporting various aspects of the pipeline:
  - **_`dataset_class.py`_** – Contains the class `EEGDataset_1D`.
  - **_`filters.py`_** – Implements `bandpass_filter`, `wavelet_denoising`, `SavGol_Wavelet`, `MNE_ICA_Wavelet`, and `SKLFast_ICA` for signal preprocessing.
  - **_`import_data.py`_** – Handles dataset loading and formatting from Iowa and San Diego sources.
  - **_`NN_classes.py`_** – Defines neural network architectures (AlexNetCustom, ResNet18).
  - **_`wavelet_transform.py`_** – Applies Wavelet Scattering Transform (WST) with Kymatio for EEG feature extraction.
- **`Data/`** – Stores the raw datasets:
  - **_`iowa/`_** – Iowa Neuroscience Institute dataset.
  - **_`san_diego/`_** – OpenNeuro ds002778 dataset with Parkinson's Disease patients.
- **`Datasets_pt/`** – Preprocessed datasets in `.pt` format (train/validation splits), ready for training.
- **`Checkpoints/`** – Saved model checkpoints and final trained models.

## 📊 Datasets Used

The project utilizes EEG data from the following sources:

1. [Iowa Neuroscience Institute](https://narayanan.lab.uiowa.edu/home/data)
2. [OpenNeuro Dataset (ds002778)](https://openneuro.org/datasets/ds002778/versions/1.0.5)

## 🔬 Models Implemented

- **AlexNetCustom** – Custom AlexNet-based architecture for EEG classification (5 convolutional layers + 3 fully connected layers), inspired by [this study](https://www.sciencedirect.com/science/article/pii/S0010482524005468).
- **ResNet18** – Pretrained ResNet18 architecture adapted for EEG classification.

Both models use **Wavelet Scattering Transform (WST)** with `J=10` and `Q=24` to convert 1D EEG signals into 2D RGB images (227×227) suitable for CNN processing.

## ⚙️ Installation & Requirements

To set up the project, ensure you have **uv** installed on your device, and CUDA 12.6 or above.

If for example you are running on a SLURM partition on a supercomputer, you may use an official Nvidia NGC Container such as [this one](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch?version=25.09-py3).

## 🚀 How to Run

### 1️⃣ Clone the repository and download the requirements

```bash
git clone https://github.com/gaspachoo/AlexNet-for-Parkinson-disease-detection-on-EEGs.git
cd AlexNet-for-Parkinson-disease-detection-on-EEGs
uv sync
```

### 2️⃣ Preprocess the EEG dataset

Apply Wavelet Scattering Transform and generate train/validation `.pt` files:

**For Iowa dataset:**

```bash
uv run python dataset_preloader.py --mode iowa --electrode AFz --filter bandpass
```

**For San Diego dataset (with medication status):**

```bash
uv run python dataset_preloader.py --mode san_diego --electrode Fz --medication off --filter wavelet
```

**Optional arguments:**

- `--segment_duration`: Segment duration in seconds (default: 5)
- `--filter`: Filtering method to apply (default: `bandpass`). Options:
  - **Mono-channel methods:**
    - `none`: No filtering (raw signal)
    - `bandpass`: Bandpass filter only (1-40 Hz)
    - `wavelet`: Bandpass + Wavelet denoising (db4, level 4)
    - `savgol`: Bandpass + SavGol-Wavelet (Savitzky-Golay + wavelet thresholding)
  - **Multi-channel ICA methods:**
    - `mne_ica`: MNE ICA with 5-criteria artifact detection + Wavelet denoising
    - `skl_ica`: sklearn FastICA with 6-criteria artifact detection

> **Note:** Multi-channel ICA methods (`mne_ica`, `skl_ica`) load ALL EEG channels for each subject, apply ICA to continuous recordings BEFORE segmentation, then extract the target electrode. This is computationally intensive but provides optimal artifact removal. The filtered datasets are saved with the filter method in the filename (e.g., `train_iowa_AFz_mne_ica.pt`).

### 3️⃣ Train the model

Train with your chosen configuration:

**Example with Iowa dataset:**

```bash
uv run python main.py --mode iowa --model alexnet --electrode AFz --filter bandpass --epochs 200 --batch_size 20 --learning_rate 1e-4 --patience 15
```

**Example with San Diego dataset:**

```bash
uv run python main.py --mode san_diego --model resnet --electrode Fz --medication off --filter wavelet --epochs 200
```

**Example with ICA-preprocessed data:**

```bash
# Train on MNE ICA filtered data
uv run python main.py --mode iowa --model alexnet --electrode AFz --filter mne_ica --epochs 200

# Train on sklearn FastICA filtered data
uv run python main.py --mode san_diego --model resnet --electrode Fz --medication off --filter skl_ica --epochs 200
```

**Available arguments:**

- `--mode`: Dataset mode (`iowa` or `san_diego`)
- `--model`: Model architecture (`alexnet` or `resnet`)
- `--electrode`: Electrode name (e.g., `AFz`, `Fz`)
- `--medication`: Medication status for San Diego (`on` or `off`, default: `on`)
- `--filter`: Filtering method used in preprocessing (default: `bandpass`). Must match the `--filter` used with `dataset_preloader.py`. Options: `none`, `bandpass`, `wavelet`, `savgol`, `mne_ica`, `skl_ica`
- `--epochs`: Maximum training epochs (default: 200)
- `--batch_size`: Batch size (default: 20)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--patience`: Early stopping patience (default: 15)

> **Important:** The `--filter` argument must match the filtering method used when preprocessing the dataset with `dataset_preloader.py`. For example, if you preprocessed with `--filter mne_ica`, you must train with `--filter mne_ica`.

### 4️⃣ Compare filtering techniques (optional)

Visualize and compare different EEG filtering techniques on raw signals:

**For Iowa dataset:**

```bash
# Format: --subject "group_idx,subject_idx" where group_idx: 0=PD, 1=Control
uv run python compare_filtering.py --dataset iowa --electrode AFz --subject "0,5" --duration 10
```

**For San Diego dataset:**

```bash
# Healthy control
uv run python compare_filtering.py --dataset san_diego --electrode Fz --subject hc1 --session hc --duration 10

# PD patient OFF medication
uv run python compare_filtering.py --dataset san_diego --electrode Fz --subject pd11 --session off --duration 10
```

**Available filtering techniques compared:**

All techniques start with bandpass filtering (1-40 Hz) for consistency:

- **Raw Signal** (unfiltered)
- **Bandpass Filter** (1-40 Hz) - mono-channel
- **Bandpass + Wavelet** (db4, level 4 wavelet denoising) - mono-channel
- **Bandpass + SavGol-Wavelet** (Savitzky-Golay detrending + wavelet thresholding) - mono-channel
- **MNE ICA + Wavelet** (MNE ICA with 5-criteria artifact detection + wavelet) - multi-channel
- **SKL Fast ICA** (sklearn FastICA with 6-criteria artifact detection) - multi-channel

> **Note:** Multi-channel techniques (MNE ICA + Wavelet, SKL Fast ICA) load all EEG channels and apply ICA across them for optimal artifact removal, then extract the target channel for visualization.

**Optional arguments:**

- `--duration`: Duration to display in seconds (default: 10)

### 5️⃣ Visualize time-frequency representations (optional)

Plot time-frequency representations:

```bash
uv run python tfr_plotter.py
```

## 📈 Results & Performance Analysis

Training results are automatically logged with **Weights & Biases (wandb)** and include:

- Training/validation loss curves
- F1-score (macro) metrics
- Confusion matrices visualization
- Model checkpoints saved in `Checkpoints/`

A confusion matrix is generated at the end of the validation step.

## 🎯 Key Features

- ✅ **Reproducible training** with fixed random seeds (`seed=42`)
- ✅ **CLI-based workflow** using `argparse` for all scripts
- ✅ **Modular architecture** separating data preprocessing, training logic, and visualization
- ✅ **Filtering comparison tool** to visualize and compare multiple signal processing techniques
- ✅ **Early stopping** to prevent overfitting
- ✅ **Automatic checkpointing** for model recovery
- ✅ **Multi-dataset support** (Iowa, San Diego)
- ✅ **Multiple architectures** (AlexNet, ResNet18)

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## 📜 License

This project is open-source under the MIT License.
