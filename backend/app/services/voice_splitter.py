import os
import logging
import subprocess
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import soundfile as sf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import librosa
import parselmouth

from app.core.config import settings
from app.services.audio_cleaner import (
    get_ffmpeg_binary,
    strip_silence_vad,
    normalize_gain,
    segment_into_uniform_chunks,
    purge_file,
    AudioCleaningError,
    InsufficientVoicedAudioError
)

logger = logging.getLogger(__name__)

@dataclass
class SpeakerAudioProfile:
    speaker_id: str
    channel_index: Optional[int]
    chunks: List[np.ndarray]
    voiced_duration_sec: float
    total_duration_sec: float
    chunk_timestamps: List[Dict[str, float]] = field(default_factory=list)
    cluster_id: Optional[int] = None
    separation_mode: str = "mono_single" # "stereo_channels" | "mono_clustered" | "mono_single"

def transcode_preserving_channels(input_path: str, output_path: str, sample_rate: int = 16000) -> Tuple[int, float]:
    """
    Transcodes input audio to 16 kHz PCM WAV, preserving original channel count (stereo or mono).
    Returns (channels, duration_seconds).
    """
    ffmpeg_bin = get_ffmpeg_binary()
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", input_path,
        "-vn",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        output_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        info = sf.info(output_path)
        return info.channels, info.duration
    except Exception as e:
        logger.error(f"FFmpeg multichannel transcode error: {e}")
        raise AudioCleaningError(f"Transcoding failed for audio file: {e}")

def extract_speaker_embedding(chunk: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extracts a compact speaker-discriminative embedding from an audio chunk:
    - 13 MFCC means (formants & vocal tract envelope)
    - Praat F0 median and std (pitch baseline & register)
    - Spectral Centroid & Zero-Crossing Rate (vocal timbre & brightness)
    """
    # 1. MFCCs 1-13
    mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # 2. Parselmouth F0 Pitch
    try:
        snd = parselmouth.Sound(chunk, sampling_frequency=sr)
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0)
        f0_vals = pitch.selected_array["frequency"]
        voiced = f0_vals[f0_vals > 0]
        if len(voiced) > 5:
            f0_med = float(np.median(voiced))
            f0_std = float(np.std(voiced))
        else:
            f0_med = 0.0
            f0_std = 0.0
    except Exception:
        f0_med = 0.0
        f0_std = 0.0

    # 3. Spectral Centroid & ZCR
    sc = librosa.feature.spectral_centroid(y=chunk, sr=sr)[0]
    sc_mean = float(np.mean(sc))
    
    zcr = librosa.feature.zero_crossing_rate(y=chunk)[0]
    zcr_mean = float(np.mean(zcr))
    
    # Combine into normalized embedding
    emb = np.concatenate([
        mfcc_mean,
        np.array([f0_med / 100.0, f0_std / 50.0, sc_mean / 1000.0, zcr_mean * 10.0])
    ])
    return np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

def split_stereo_speakers(
    data_int16: np.ndarray,
    sr: int = 16000,
    total_duration_sec: float = 0.0
) -> List[SpeakerAudioProfile]:
    """
    Splits two-channel (stereo) audio call recordings into distinct speaker channels.
    Typically Channel 0 = Local Agent / Caller, Channel 1 = Remote Party.
    """
    profiles = []
    channel_labels = ["Voice 1 (Channel L)", "Voice 2 (Channel R)"]
    
    for ch_idx in range(2):
        channel_data = data_int16[:, ch_idx]
        
        voiced_int16, voiced_dur, _ = strip_silence_vad(
            channel_data,
            sample_rate=sr,
            frame_duration_ms=settings.VAD_FRAME_MS,
            aggressiveness=settings.VAD_AGGRESSIVENESS
        )
        
        if voiced_dur < 0.5:
            # Fall back to raw channel data if VAD stripped too aggressively
            voiced_float = channel_data.astype(np.float32) / 32768.0
            voiced_dur = total_duration_sec
        else:
            voiced_float = voiced_int16.astype(np.float32) / 32768.0

        normalized = normalize_gain(voiced_float, target_dbfs=-20.0)
        chunks = segment_into_uniform_chunks(normalized, sample_rate=sr, chunk_duration_sec=settings.CHUNK_DURATION_SEC)
        
        chunk_ts = []
        for i in range(len(chunks)):
            st = i * settings.CHUNK_DURATION_SEC
            chunk_ts.append({
                "start_time_sec": round(st, 2),
                "end_time_sec": round(st + settings.CHUNK_DURATION_SEC, 2)
            })

        profiles.append(SpeakerAudioProfile(
            speaker_id=channel_labels[ch_idx],
            channel_index=ch_idx,
            chunks=chunks,
            voiced_duration_sec=round(voiced_dur, 2),
            total_duration_sec=round(total_duration_sec, 2),
            chunk_timestamps=chunk_ts,
            separation_mode="stereo_channels"
        ))
        
    return profiles

def cluster_mono_speakers(
    data_int16: np.ndarray,
    sr: int = 16000,
    total_duration_sec: float = 0.0
) -> List[SpeakerAudioProfile]:
    """
    Diarizes mixed single-channel audio into distinct speaker profiles using
    feature extraction and KMeans clustering.
    """
    # 1. Silence stripping via VAD
    voiced_int16, voiced_dur, _ = strip_silence_vad(
        data_int16,
        sample_rate=sr,
        frame_duration_ms=settings.VAD_FRAME_MS,
        aggressiveness=settings.VAD_AGGRESSIVENESS
    )
    
    if voiced_dur < settings.MIN_VOICED_DURATION_SEC:
        voiced_float = data_int16.astype(np.float32) / 32768.0
        voiced_dur = total_duration_sec
    else:
        voiced_float = voiced_int16.astype(np.float32) / 32768.0
        
    normalized = normalize_gain(voiced_float, target_dbfs=-20.0)
    
    # 2. Segment into uniform 4-second sub-windows
    chunks = segment_into_uniform_chunks(normalized, sample_rate=sr, chunk_duration_sec=settings.CHUNK_DURATION_SEC)
    
    if len(chunks) < 3:
        # Not enough temporal windows for clustering; treat as single speaker
        chunk_ts = [
            {"start_time_sec": round(i * settings.CHUNK_DURATION_SEC, 2),
             "end_time_sec": round((i + 1) * settings.CHUNK_DURATION_SEC, 2)}
            for i in range(len(chunks))
        ]
        return [SpeakerAudioProfile(
            speaker_id="Voice 1",
            channel_index=None,
            chunks=chunks,
            voiced_duration_sec=round(voiced_dur, 2),
            total_duration_sec=round(total_duration_sec, 2),
            chunk_timestamps=chunk_ts,
            separation_mode="mono_single"
        )]

    # 3. Extract speaker embeddings per chunk
    embeddings = [extract_speaker_embedding(c, sr=sr) for c in chunks]
    X = np.array(embeddings)
    
    # Scale embeddings
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Perform KMeans Clustering (k=2)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Check cluster quality
    sil_score = -1.0
    try:
        sil_score = silhouette_score(X_scaled, labels)
    except Exception:
        pass

    logger.info(f"KMeans Diarization: {len(chunks)} chunks, silhouette score: {sil_score:.3f}")

    # If silhouette score is low, audio is homogeneous (single speaker)
    if sil_score < 0.12 or np.sum(labels == 0) == 0 or np.sum(labels == 1) == 0:
        chunk_ts = [
            {"start_time_sec": round(i * settings.CHUNK_DURATION_SEC, 2),
             "end_time_sec": round((i + 1) * settings.CHUNK_DURATION_SEC, 2)}
            for i in range(len(chunks))
        ]
        return [SpeakerAudioProfile(
            speaker_id="Voice 1",
            channel_index=None,
            chunks=chunks,
            voiced_duration_sec=round(voiced_dur, 2),
            total_duration_sec=round(total_duration_sec, 2),
            chunk_timestamps=chunk_ts,
            separation_mode="mono_single"
        )]

    # 5. Group into two distinct speaker profiles
    profiles = []
    for cluster_id, label_name in [(0, "Voice 1"), (1, "Voice 2")]:
        cluster_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        cluster_chunks = [chunks[i] for i in cluster_indices]
        cluster_ts = [
            {"start_time_sec": round(i * settings.CHUNK_DURATION_SEC, 2),
             "end_time_sec": round((i + 1) * settings.CHUNK_DURATION_SEC, 2)}
            for i in cluster_indices
        ]
        cluster_dur = len(cluster_chunks) * settings.CHUNK_DURATION_SEC
        
        profiles.append(SpeakerAudioProfile(
            speaker_id=label_name,
            channel_index=None,
            chunks=cluster_chunks,
            voiced_duration_sec=round(cluster_dur, 2),
            total_duration_sec=round(total_duration_sec, 2),
            chunk_timestamps=cluster_ts,
            cluster_id=cluster_id,
            separation_mode="mono_clustered"
        ))
        
    return profiles

def separate_speakers(
    file_path: str,
    temp_dir: Optional[str] = None
) -> Tuple[List[SpeakerAudioProfile], Dict[str, Any]]:
    """
    Main Multi-Speaker Separation Engine:
    1. Transcodes audio preserving channels.
    2. If stereo (2 channels): splits channels separately.
    3. If mono (1 channel): clusters speech chunks into speaker profiles via KMeans.
    4. Purges intermediate audio safely.
    """
    target_dir = temp_dir or str(settings.TEMP_DIR)
    os.makedirs(target_dir, exist_ok=True)
    temp_wav = os.path.join(target_dir, f"split_transcode_{os.path.basename(file_path)}.wav")
    
    try:
        channels, total_dur = transcode_preserving_channels(file_path, temp_wav, sample_rate=settings.SAMPLE_RATE)
        data_int16, sr = sf.read(temp_wav, dtype="int16")
        
        if channels == 2 and data_int16.ndim == 2:
            logger.info("Stereo audio detected (2 channels). Processing channels separately...")
            profiles = split_stereo_speakers(data_int16, sr=sr, total_duration_sec=total_dur)
            mode = "stereo_channels"
        else:
            # Flatten to 1D if single channel
            mono_data = data_int16[:, 0] if data_int16.ndim == 2 else data_int16
            logger.info("Single-channel audio detected. Running KMeans speech diarization...")
            profiles = cluster_mono_speakers(mono_data, sr=sr, total_duration_sec=total_dur)
            mode = "mono_clustered" if len(profiles) > 1 else "mono_single"

        telemetry = {
            "channels_detected": channels,
            "speakers_identified": len(profiles),
            "separation_mode": mode,
            "total_duration_sec": round(float(total_dur), 2)
        }
        return profiles, telemetry
        
    finally:
        purge_file(temp_wav)
