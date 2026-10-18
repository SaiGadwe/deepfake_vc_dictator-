import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import scipy.stats as stats
import scipy.fftpack as fftpack
import librosa
import parselmouth

logger = logging.getLogger(__name__)

def extract_lfcc(
    audio: np.ndarray,
    sr: int = 16000,
    n_lfcc: int = 20,
    n_filters: int = 24,
    n_fft: int = 512,
    hop_length: int = 256
) -> np.ndarray:
    """
    Computes Linear Frequency Cepstral Coefficients (LFCCs).
    Unlike MFCCs (which compress high frequencies via the Mel scale), LFCCs apply
    linearly spaced filterbanks, making them exceptionally sensitive to high-frequency
    artifacts and phase inconsistencies introduced by neural vocoders (HiFi-GAN, WaveGlow, etc.).
    
    Returns:
        lfcc: 2D array of shape (num_frames, n_lfcc)
    """
    pad_width = n_fft // 2
    padded = np.pad(audio, (pad_width, pad_width), mode="reflect")
    
    # Efficient sliding window STFT power spectrum
    windows = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_length]
    windowed = windows * np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(windowed, axis=-1)) ** 2  # (frames, n_fft//2 + 1)
    
    # Linear filterbank calculation
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_freqs)
    filter_freqs = np.linspace(0, sr / 2, n_filters + 2)
    
    fb = np.zeros((n_filters, n_freqs), dtype=np.float32)
    for m in range(1, n_filters + 1):
        f_left = filter_freqs[m - 1]
        f_center = filter_freqs[m]
        f_right = filter_freqs[m + 1]
        for k in range(n_freqs):
            f = freqs[k]
            if f_left <= f <= f_center and f_center > f_left:
                fb[m - 1, k] = (f - f_left) / (f_center - f_left)
            elif f_center < f <= f_right and f_right > f_center:
                fb[m - 1, k] = (f_right - f) / (f_right - f_center)
                
    # Filterbank energies
    fb_energy = np.dot(spec, fb.T)
    fb_energy = np.where(fb_energy <= 1e-12, 1e-12, fb_energy)
    log_fb_energy = np.log10(fb_energy)
    
    # Discrete Cosine Transform (DCT Type-II)
    lfcc = fftpack.dct(log_fb_energy, type=2, axis=-1, norm="ortho")[:, :n_lfcc]
    return lfcc

def extract_pitch_parselmouth(
    audio: np.ndarray,
    sr: int = 16000,
    time_step: float = 0.01,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 600.0
) -> Dict[str, float]:
    """
    Extracts high-precision Fundamental Frequency (F0), Jitter, and Shimmer
    using the gold-standard Praat acoustic engine (via Parselmouth).
    
    Synthetic voices frequently display unnatural pitch flatness, discrete pitch quantization,
    or lack of organic micro-prosodic perturbations.
    """
    try:
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        pitch = sound.to_pitch(time_step=time_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
        f0_values = pitch.selected_array["frequency"]
        voiced = f0_values[f0_values > 0]
        
        total_frames = len(f0_values)
        voiced_frames = len(voiced)
        voiced_fraction = voiced_frames / (total_frames + 1e-6)
        
        if voiced_frames > 5:
            f0_mean = float(np.mean(voiced))
            f0_std = float(np.std(voiced))
            f0_min = float(np.min(voiced))
            f0_max = float(np.max(voiced))
            f0_range = f0_max - f0_min
            
            # Pitch entropy (distribution spread)
            hist, _ = np.histogram(voiced, bins=10, density=True)
            hist = hist[hist > 0]
            f0_entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))
            
            # Jitter and Shimmer estimates
            diffs = np.abs(np.diff(voiced))
            jitter_local = float(np.mean(diffs) / (f0_mean + 1e-6))
        else:
            f0_mean = 0.0
            f0_std = 0.0
            f0_min = 0.0
            f0_max = 0.0
            f0_range = 0.0
            f0_entropy = 0.0
            jitter_local = 0.0

        # Shimmer estimate via amplitude envelope differences
        env = np.abs(audio)
        env_peaks = env[env > np.mean(env)]
        if len(env_peaks) > 10:
            shimmer_local = float(np.mean(np.abs(np.diff(env_peaks))) / (np.mean(env_peaks) + 1e-6))
        else:
            shimmer_local = 0.0

        return {
            "f0_mean": f0_mean,
            "f0_std": f0_std,
            "f0_min": f0_min,
            "f0_max": f0_max,
            "f0_range": f0_range,
            "f0_entropy": f0_entropy,
            "voiced_fraction": voiced_fraction,
            "jitter_local": jitter_local,
            "shimmer_local": shimmer_local,
        }
    except Exception as e:
        logger.warning(f"Parselmouth pitch extraction failed: {e}")
        return {
            "f0_mean": 0.0,
            "f0_std": 0.0,
            "f0_min": 0.0,
            "f0_max": 0.0,
            "f0_range": 0.0,
            "f0_entropy": 0.0,
            "voiced_fraction": 0.0,
            "jitter_local": 0.0,
            "shimmer_local": 0.0,
        }

def extract_spectral_loudness_features(audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """
    Computes spectral centroid, spectral rolloff, spectral flatness,
    RMS loudness, and zero crossing statistics over the chunk.
    """
    # 1. Spectral Centroid
    sc = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    sc_mean = float(np.mean(sc))
    sc_std = float(np.std(sc))
    sc_skew = float(stats.skew(sc)) if len(sc) > 3 else 0.0
    
    # 2. Spectral Rolloff (85% energy roll-off)
    rf = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)[0]
    rf_mean = float(np.mean(rf))
    rf_std = float(np.std(rf))
    
    # 3. Spectral Flatness
    sf = librosa.feature.spectral_flatness(y=audio)[0]
    sf_mean = float(np.mean(sf))
    sf_std = float(np.std(sf))
    
    # 4. RMS Loudness / Energy
    rms = librosa.feature.rms(y=audio)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    rms_peak = float(np.max(rms)) if len(rms) > 0 else 0.0
    peak_to_rms = rms_peak / (rms_mean + 1e-6)
    
    # 5. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))
    
    return {
        "spectral_centroid_mean": sc_mean,
        "spectral_centroid_std": sc_std,
        "spectral_centroid_skew": sc_skew,
        "spectral_rolloff_mean": rf_mean,
        "spectral_rolloff_std": rf_std,
        "spectral_flatness_mean": sf_mean,
        "spectral_flatness_std": sf_std,
        "rms_loudness_mean": rms_mean,
        "rms_loudness_std": rms_std,
        "peak_to_rms_ratio": peak_to_rms,
        "zcr_mean": zcr_mean,
        "zcr_std": zcr_std
    }

def extract_chunk_features(
    chunk: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 40,
    n_lfcc: int = 20
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Extracts the combined 201-dimensional feature vector for a uniform 4-second audio window.
    
    Returns:
        feature_vector: 1D numpy array of shape (201,)
        feature_names: List of string names for every feature index
        raw_telemetry: Diagnostic summary dictionary
    """
    feature_names = []
    feature_values = []
    
    # 1. MFCCs (1–40) + statistics (mean, variance, skew)
    mfcc_matrix = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc_matrix, axis=1)
    mfcc_var = np.var(mfcc_matrix, axis=1)
    mfcc_skew = stats.skew(mfcc_matrix, axis=1)
    
    for i in range(n_mfcc):
        feature_names.append(f"mfcc_{i+1}_mean")
        feature_values.append(float(mfcc_mean[i]))
    for i in range(n_mfcc):
        feature_names.append(f"mfcc_{i+1}_var")
        feature_values.append(float(mfcc_var[i]))
    for i in range(n_mfcc):
        feature_names.append(f"mfcc_{i+1}_skew")
        val = float(mfcc_skew[i])
        feature_values.append(val if np.isfinite(val) else 0.0)
        
    # 2. LFCCs (1–20) + statistics (mean, variance, skew)
    lfcc_matrix = extract_lfcc(chunk, sr=sr, n_lfcc=n_lfcc)
    lfcc_mean = np.mean(lfcc_matrix, axis=0)
    lfcc_var = np.var(lfcc_matrix, axis=0)
    lfcc_skew = stats.skew(lfcc_matrix, axis=0)
    
    for i in range(n_lfcc):
        feature_names.append(f"lfcc_{i+1}_mean")
        feature_values.append(float(lfcc_mean[i]))
    for i in range(n_lfcc):
        feature_names.append(f"lfcc_{i+1}_var")
        feature_values.append(float(lfcc_var[i]))
    for i in range(n_lfcc):
        feature_names.append(f"lfcc_{i+1}_skew")
        val = float(lfcc_skew[i])
        feature_values.append(val if np.isfinite(val) else 0.0)
        
    # 3. Parselmouth Pitch (F0), Jitter, Shimmer
    pitch_features = extract_pitch_parselmouth(chunk, sr=sr)
    for k, v in pitch_features.items():
        feature_names.append(k)
        feature_values.append(float(v))
        
    # 4. Spectral & Loudness summary statistics
    spectral_features = extract_spectral_loudness_features(chunk, sr=sr)
    for k, v in spectral_features.items():
        feature_names.append(k)
        feature_values.append(float(v))
        
    feature_vector = np.array(feature_values, dtype=np.float32)
    # Sanitize any NaNs or infinities
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    
    raw_telemetry = {
        "f0_mean_hz": round(pitch_features.get("f0_mean", 0.0), 1),
        "f0_std_hz": round(pitch_features.get("f0_std", 0.0), 2),
        "voiced_fraction": round(pitch_features.get("voiced_fraction", 0.0), 2),
        "spectral_centroid_hz": round(spectral_features.get("spectral_centroid_mean", 0.0), 1),
        "rms_loudness": round(spectral_features.get("rms_loudness_mean", 0.0), 4)
    }
    
    return feature_vector, feature_names, raw_telemetry

def get_feature_names() -> List[str]:
    """Helper returning standard list of feature names."""
    dummy = np.zeros(64000, dtype=np.float32)
    _, names, _ = extract_chunk_features(dummy)
    return names
