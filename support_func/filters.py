import mne
import numpy as np
import pywt
from scipy.signal import butter, filtfilt, savgol_filter
from sklearn.decomposition import FastICA


def bandpass_filter(signal, lowcut=0.5, highcut=40, fs=128, order=4):
    """Band-pass filter for EEG signal."""
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, signal)


def wavelet_denoising(signal, wavelet="db4", level=4):
    """Wavelet denoising with adaptive thresholding."""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Noise estimation
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    denoised_coeffs = [coeffs[0]]  # Keep approximation coefficients
    for detail_coeff in coeffs[1:]:
        denoised_detail = pywt.threshold(detail_coeff, threshold, mode="soft")
        denoised_coeffs.append(denoised_detail)

    return pywt.waverec(denoised_coeffs, wavelet)


def modern_cleaning(eeg_data, sfreq=128):
    """
    Optimized EEG cleaning: band-pass filter, ICA with robust artifact exclusion, and wavelet denoising.

    Parameters:
        eeg_data: np.ndarray (channels, samples)
        sfreq: Sampling frequency (Hz)

    Returns:
        cleaned_data: np.ndarray (channels, samples)
    """

    n_channels, n_times = eeg_data.shape
    ch_names = [f"EEG {i + 1}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_data, info)

    # Bandpass filter to remove low-frequency drift and high-frequency noise
    raw.filter(1.0, 40.0, fir_design="firwin", verbose=False)

    # Adapt n_components to the number of channels available
    n_components = min(15, n_channels)

    ica = mne.preprocessing.ICA(
        n_components=n_components, random_state=97, max_iter="auto"
    )
    ica.fit(raw, verbose=False)

    # Automatic artifact detection using multiple robust criteria
    sources = ica.get_sources(raw).get_data()

    # Compute comprehensive statistics for each component
    artifact_indices = []
    artifact_scores = []  # Track artifact likelihood for each component

    for i in range(n_components):
        source = sources[i]
        score = 0.0  # Artifact likelihood score

        # === CRITERION 1: Variance-based detection ===
        var = np.var(source)
        median_var = np.median([np.var(sources[j]) for j in range(n_components)])
        var_ratio = var / (median_var + 1e-10)
        if var_ratio > 2.0:
            score += (var_ratio - 2.0) * 0.5

        # === CRITERION 2: Kurtosis (measures spikiness) ===
        mean_s = np.mean(source)
        std_s = np.std(source)
        if std_s > 1e-10:
            kurtosis = np.mean((source - mean_s) ** 4) / (std_s**4)
            # Normal distribution has kurtosis = 3
            # Artifacts (eye blinks, muscle) typically have kurtosis > 5
            if kurtosis > 5.0:
                score += (kurtosis - 5.0) * 0.3
        else:
            kurtosis = 3.0

        # === CRITERION 3: Range/Peak-to-Peak amplitude ===
        ptp = np.ptp(source)
        median_ptp = np.median([np.ptp(sources[j]) for j in range(n_components)])
        ptp_ratio = ptp / (median_ptp + 1e-10)
        if ptp_ratio > 2.0:
            score += (ptp_ratio - 2.0) * 0.4

        # === CRITERION 4: High-frequency content ===
        # Calculate power in high-frequency band (20-40 Hz range)
        source_fft = np.fft.rfft(source)
        freqs = np.fft.rfftfreq(len(source), 1 / sfreq)
        hf_mask = (freqs >= 20) & (freqs <= 40)
        lf_mask = (freqs >= 1) & (freqs < 20)
        hf_power = np.sum(np.abs(source_fft[hf_mask]) ** 2)
        lf_power = np.sum(np.abs(source_fft[lf_mask]) ** 2)
        hf_ratio = hf_power / (lf_power + hf_power + 1e-10)
        if hf_ratio > 0.3:  # >30% high-freq power = likely muscle artifact
            score += (hf_ratio - 0.3) * 2.0

        # === CRITERION 5: Entropy (measures signal complexity) ===
        # Low entropy = simple, repetitive patterns (like heartbeat, eye blinks)
        hist, _ = np.histogram(source, bins=50)
        hist = hist / (np.sum(hist) + 1e-10)
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        if entropy < 3.0:  # Low entropy suggests artifact
            score += (3.0 - entropy) * 0.2

        artifact_scores.append(score)

        # Mark as artifact if score exceeds threshold
        if score > 1.5:  # Tuned threshold
            artifact_indices.append(i)

    ica.exclude = artifact_indices
    print(
        f"  MNE ICA excluded components: {artifact_indices} ({len(artifact_indices)}/{n_components} removed)"
    )
    # Uncomment for debugging: print(f"  Artifact scores: {[f'{s:.2f}' for s in artifact_scores]}")

    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)

    # Additional wavelet denoising on each channel
    cleaned_data = raw_clean.get_data()
    for ch in range(n_channels):
        cleaned_data[ch, :] = wavelet_denoising(cleaned_data[ch, :])

    return cleaned_data

    cleaned_data = raw_clean.get_data()
    for ch in range(n_channels):
        cleaned_data[ch, :] = wavelet_denoising(cleaned_data[ch, :])

    return cleaned_data


def matlab_like_cleaning(
    eeg_data, polyorder=5, window_length=127, wavelet="db2", level=4
):
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


def SKLFast_ICA(eeg_data, lda=2.5):
    """
    SKLearn-based ICA for artifact removal with robust multi-criteria detection.

    Parameters:
        eeg_data: np.ndarray (channels, samples)
        lda: Threshold multiplier for artifact detection (default: 2.5)

    Returns:
        cleaned_eeg: np.ndarray (channels, samples)
    """
    n_channels, n_samples = eeg_data.shape

    if n_channels != 32:
        print(f"  Warning: SKLFast_ICA optimized for 32 channels, got {n_channels}.")

    # Step 1: Apply bandpass filter to each channel (remove drift and high-freq noise)
    filtered_eeg = np.zeros_like(eeg_data)
    for ch in range(n_channels):
        filtered_eeg[ch, :] = bandpass_filter(
            eeg_data[ch, :], lowcut=0.5, highcut=40, fs=128
        )

    # Step 2: Normalize for ICA (critical for convergence)
    mean = np.mean(filtered_eeg, axis=1, keepdims=True)
    std = np.std(filtered_eeg, axis=1, keepdims=True)
    normalized_eeg = (filtered_eeg - mean) / (std + 1e-10)

    # Step 3: Apply ICA
    ica = FastICA(n_components=n_channels, random_state=42, max_iter=1000, tol=0.001)
    ica_sources = ica.fit_transform(normalized_eeg.T).T

    # Step 4: Identify artifacts using robust scoring system
    artifact_indices = []
    artifact_scores = []

    # Pre-compute global statistics
    all_vars = [np.var(ica_sources[i]) for i in range(n_channels)]
    median_var = np.median(all_vars)
    all_ptp = [np.ptp(ica_sources[i]) for i in range(n_channels)]
    median_ptp = np.median(all_ptp)

    # Compute sampling frequency (assume 128 Hz if not provided)
    fs = 128

    for i in range(n_channels):
        source = ica_sources[i]
        score = 0.0

        # === CRITERION 1: Abnormally high variance ===
        var_ratio = all_vars[i] / (median_var + 1e-10)
        if var_ratio > 2.0:
            score += (var_ratio - 2.0) * 0.4

        # === CRITERION 2: High peak-to-peak amplitude ===
        ptp_ratio = all_ptp[i] / (median_ptp + 1e-10)
        if ptp_ratio > 2.0:
            score += (ptp_ratio - 2.0) * 0.5

        # === CRITERION 3: Kurtosis (spikiness) ===
        mean_s = np.mean(source)
        std_s = np.std(source)
        if std_s > 1e-10:
            kurtosis = np.mean((source - mean_s) ** 4) / (std_s**4)
            if kurtosis > 5.0:
                score += (kurtosis - 5.0) * 0.3
        else:
            kurtosis = 3.0

        # === CRITERION 4: High-frequency power ===
        # FFT-based frequency analysis
        source_fft = np.fft.rfft(source)
        freqs = np.fft.rfftfreq(len(source), 1 / fs)
        hf_mask = (freqs >= 20) & (freqs <= 40)
        lf_mask = (freqs >= 1) & (freqs < 20)
        hf_power = np.sum(np.abs(source_fft[hf_mask]) ** 2)
        lf_power = np.sum(np.abs(source_fft[lf_mask]) ** 2)
        hf_ratio = hf_power / (lf_power + hf_power + 1e-10)
        if hf_ratio > 0.25:  # >25% high-freq = muscle artifact
            score += (hf_ratio - 0.25) * 2.5

        # === CRITERION 5: Low-frequency power (eye movements, drift) ===
        vlf_mask = freqs < 1.0
        vlf_power = np.sum(np.abs(source_fft[vlf_mask]) ** 2)
        vlf_ratio = vlf_power / (np.sum(np.abs(source_fft) ** 2) + 1e-10)
        if vlf_ratio > 0.15:  # >15% very low freq = drift/eye movement
            score += (vlf_ratio - 0.15) * 1.5

        # === CRITERION 6: Signal entropy ===
        hist, _ = np.histogram(source, bins=50)
        hist = hist / (np.sum(hist) + 1e-10)
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        if entropy < 3.2:
            score += (3.2 - entropy) * 0.15

        artifact_scores.append(score)

        # Mark as artifact if score exceeds threshold
        if score > 1.2:  # Calibrated threshold
            artifact_indices.append(i)

    print(
        f"  SKL ICA identified artifacts: {artifact_indices} ({len(artifact_indices)}/{n_channels} removed)"
    )
    # Uncomment for debugging: print(f"  Artifact scores: {[f'{s:.2f}' for s in artifact_scores]}")

    # Step 5: Zero out artifact components
    ica_sources[artifact_indices, :] = 0

    # Step 6: Reconstruct and denormalize
    cleaned_normalized = ica.inverse_transform(ica_sources.T).T
    cleaned_eeg = cleaned_normalized * std + mean

    # Step 7: Additional wavelet denoising (like MATLAB method)
    for ch in range(n_channels):
        cleaned_eeg[ch, :] = wavelet_denoising(cleaned_eeg[ch, :])

    return cleaned_eeg
