import mne
import numpy as np
import pywt
from scipy.signal import butter, filtfilt, savgol_filter
from sklearn.decomposition import FastICA

def bandpass_filter(signal, lowcut=0.5, highcut=40, fs=128, order=4):
    """Band-pass filter for EEG signal."""
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    return filtfilt(b, a, signal)


def wavelet_denoising(signal, wavelet='db4', level=4):
    """Wavelet denoising with adaptive thresholding."""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Noise estimation
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    denoised_coeffs = [coeffs[0]]  # Keep approximation coefficients
    for detail_coeff in coeffs[1:]:
        denoised_detail = pywt.threshold(detail_coeff, threshold, mode='soft')
        denoised_coeffs.append(denoised_detail)

    return pywt.waverec(denoised_coeffs, wavelet)

def modern_cleaning(eeg_data, sfreq=128):
    """
    Optimized EEG cleaning: band-pass filter, ICA, and wavelet denoising.
    
    eeg_data: np.ndarray (channels, samples)
    sfreq: Sampling frequency (Hz)
    
    Returns:
        cleaned_data: np.ndarray (channels, samples)
    """

    n_channels, n_times = eeg_data.shape
    ch_names = [f"EEG {i+1}" for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_data, info)

    raw.filter(1., 40., fir_design='firwin')

    ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter='auto')
    ica.fit(raw)

    # Comment out automatic artifact detection (no EOG channels)
    # eog_inds, scores = ica.find_bads_eog(raw)
    # ecg_inds, scores = ica.find_bads_ecg(raw)
    # ica.exclude = eog_inds + ecg_inds

    # Optional: manual exclusion after visualization
    # ica.plot_components()
    # ica.exclude = [0, 1]  # Example if needed

    raw_clean = raw.copy()
    ica.apply(raw_clean)

    cleaned_data = raw_clean.get_data()
    for ch in range(n_channels):
        cleaned_data[ch, :] = wavelet_denoising(cleaned_data[ch, :])

    return cleaned_data


def matlab_like_cleaning(eeg_data, polyorder=5, window_length=127, wavelet='db2', level=4):
    """
    Translation of MATLAB EEG cleaning using Savitzky-Golay filter and wavelet thresholding.
    
    Parameters:
        eeg_data: np.ndarray, shape (channels, samples)
        polyorder: int, polynomial order for Savitzky-Golay filter
        window_length: int, window length for Savitzky-Golay filter
        wavelet: str, wavelet type for decomposition
        level: int, decomposition level
    
    Returns:
        clean_data: np.ndarray, cleaned EEG signals
    """
    num_channels, num_samples = eeg_data.shape
    cancelled = np.zeros_like(eeg_data)
    clean_data = np.zeros_like(eeg_data)

    # Step 1: Subtracting the trend using Savitzky-Golay filter
    for ch in range(num_channels):
        primary = eeg_data[ch, :]
        trend = savgol_filter(primary, window_length, polyorder)
        cancelled[ch, :] = primary - trend

    # Step 2: Wavelet thresholding
    for ch in range(num_channels):
        # Wavelet decomposition
        coeffs = pywt.wavedec(cancelled[ch, :], wavelet, level=level)
        approx = coeffs[0]
        details = coeffs[1:]

        # Threshold based on the standard deviation of detail coefficients (level 3)
        t = np.std(details[2]) * 0.8

        # Apply thresholding to approximation and details
        approx = np.sign(approx) * np.minimum(np.abs(approx), t)
        details = [np.sign(cd) * np.minimum(np.abs(cd), t) for cd in details]

        # Reconstruct the signal
        coeffs = [approx] + details
        clean = pywt.waverec(coeffs, wavelet)
        
        # Truncate to original length in case of padding during wavelet processing
        clean_data[ch, :] = clean[:num_samples]

    return clean_data


def SKLFast_ICA(eeg_data, lda=6):
    filtered_eeg = bandpass_filter(eeg_data)
    n_channels = eeg_data.shape[0]
    if n_channels == 32:
        pass
    else:
        print('error')
    # Apply ICA (Independent Component Analysis)
    ica = FastICA(n_components=n_channels, random_state=42)
    ica_sources = ica.fit_transform(filtered_eeg.T).T

    # Identify Artifacts (Heuristic: High Amplitude Components)
    artifact_indices = []
    for i in range(n_channels):
        if np.max(np.abs(ica_sources[i])) > lda * np.median(np.abs(ica_sources[i])):  # Threshold-based detection
            artifact_indices.append(i)

    print(f"Identified Artifact Components: {artifact_indices}")

    # Remove Artifacts by Zeroing Out Identified Components
    ica_sources[artifact_indices, :] = 0

    # Reconstruct EEG Data (Inverse ICA)
    cleaned_eeg = ica.inverse_transform(ica_sources.T).T
    
    return cleaned_eeg