# 🧠 PD Detection framework : Applying WST and AlexNet on EEGs 

## 📌 Project Overview
This project focuses on implementing and evaluating different neural network architectures for EEG classification using **PyTorch**. It EEG datasets in order to be able to recognize if a patient is suffering from PD thanks to a simple EEG of a few minutes.

## 📂 Project Structure
The repository is organized as follows:
   
- **`dataset_preloader.py`** – Automates dataset preprocessing and loading.  
- **`main.py`** – Entry point to execute the training and evaluation process.  
- **`tfr_plotter.py`** – Generates visualizations for EEG signal transformations.  
- **`support_func/`** – Contains utility functions supporting various aspects of the pipeline:
   - ***`dataset_class.py`*** – Contains the class `EEGDataset_1D`.
   - ***`filters.py`*** – Contains the `bandpass_filter`.  
   - ***`import_data.py`*** – Handles dataset loading and formatting.
   - ***`NN_classes.py`*** – Defines the neural network architectures, including AlexNet-based models.
   - ***`wavelet_transform.py`*** – Applies wavelet transformation for EEG feature extraction.  
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
To set up the project, ensure you have the following dependencies installed:

```bash
pip install h5py kymatio matplotlib mne numpy scikit-learn scipy seaborn torchmetrics
```
Also install torch, torchvision, torchaudio compatible with your cuda version
## 🚀 How to Run

1. Clone the repository:  
   ```bash
   git clone https://github.com/gaspachoo/Internship-MU-Assignment-2.git
   cd your-repo-name
   ```
2. Preprocess the EEG dataset:  
   ```bash
   python data_set_preloader.py
   ```
3. Train the model:  
   ```bash
   python main.py
   ```
4. Plot data:  
   ```bash
   tfr_plotter.py
   ```

## 📈 Results & Performance Analysis
Results are plot after executing **`main.py`**, including confusion matrices and classification reports.  
The impact of different preprocessing techniques (e.g., wavelet transform, filtering) is analyzed in **`tfr_plotter.py`**.

## 🤝 Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

## 📜 License
This project is open-source under the MIT License.
