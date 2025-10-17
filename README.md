# 🧠 PD Detection framework : Applying WST and AlexNet on EEGs

## 📌 Project Overview

This project focuses on implementing and evaluating different neural network architectures for EEG classification using **PyTorch**. It EEG datasets in order to be able to recognize if a patient is suffering from PD thanks to a simple EEG of a few minutes.

## 📂 Project Structure

The repository is organized as follows:

- **`dataset_preloader.py`** – Automates dataset preprocessing and loading.
- **`main.py`** – Entry point to execute the training and evaluation process.
- **`tfr_plotter.py`** – Generates visualizations for EEG signal transformations.
- **`support_func/`** – Contains utility functions supporting various aspects of the pipeline:
  - **_`dataset_class.py`_** – Contains the class `EEGDataset_1D`.
  - **_`filters.py`_** – Contains the `bandpass_filter`.
  - **_`import_data.py`_** – Handles dataset loading and formatting.
  - **_`NN_classes.py`_** – Defines the neural network architectures, including AlexNet-based models.
  - **_`wavelet_transform.py`_** – Applies wavelet transformation for EEG feature extraction.
- **`Data/`** – Stores the datasets source folders used for preparing the training and testing.
- **`Datasets_pt/`** – Stores the datasets in .pt format, ready to be used for training and testing.

## 📊 Datasets Used

The project utilizes EEG data from the following sources:

1. [Iowa Neuroscience Institute](https://narayanan.lab.uiowa.edu/home/data)
2. [OpenNeuro Dataset (ds002778)](https://openneuro.org/datasets/ds002778/versions/1.0.5)

## 🔬 Checkpoints Implemented

- **AlexNet-based architecture** for EEG classification, inspired by [this study](https://www.sciencedirect.com/science/article/pii/S0010482524005468).
- Additional baseline models (e.g., fully connected networks).

## ⚙️ Installation & Requirements

To set up the project, ensure you have **uv** installed on your device, and CUDA 12.6 or above.

If for example you are running on a SLURM partition on a supercomputer, you may use an official Nvidia NGC Container such as [this one](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch?version=25.09-py3).

Afterwards, you can then run :

```bash
uv sync
```

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/gaspachoo/AlexNet-for-Parkinson-disease-detection-on-EEGs.git
   cd AlexNet-for-Parkinson-disease-detection-on-EEGs
   ```
2. Preprocess the EEG dataset:
   ```bash
   uv run dataset_preloader.py
   ```
3. Train the model:
   ```bash
   uv run main.py
   ```
4. Plot data:
   ```bash
   uv run tfr_plotter.py
   ```

## 📈 Results & Performance Analysis

Results are plot after executing **`main.py`**, including confusion matrices and classification reports.  
The impact of different preprocessing techniques (e.g., wavelet transform, filtering) is analyzed in **`tfr_plotter.py`**.

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## 📜 License

This project is open-source under the MIT License.
