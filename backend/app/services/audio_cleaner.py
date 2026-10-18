import os
import shutil
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import soundfile as sf
import webrtcvad
import imageio_ffmpeg
from app.core.config import settings

logger = logging.getLogger(__name__)

class AudioCleaningError(Exception):
    """Raised when audio conversion or preprocessing fails."""
    pass

class InsufficientVoicedAudioError(AudioCleaningError):
    """Raised when audio contains little to no voiced speech after VAD."""
    pass

def get_ffmpeg_binary() -> str:
    """
    Locates the FFmpeg executable from system PATH or bundled imageio_ffmpeg.
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and os.path.exists(bundled):
            return bundled
    except Exception as e:
        logger.warning(f"Could not load imageio_ffmpeg executable: {e}")
    
    raise AudioCleaningError("FFmpeg binary not found in system PATH or bundled packages.")

def convert_to_standard_wav(input_path: str, output_path: str, sample_rate: int = 16000) -> str:
    """
    Converts any incoming audio format (MP3, OGG, WAV, M4A, etc.) to
    standard 16 kHz single-channel (mono) 16-bit PCM WAV using FFmpeg.
    """
    ffmpeg_bin = get_ffmpeg_binary()
    cmd = [
        ffmpeg_bin,
        "-y",               # Overwrite output
        "-i", input_path,   # Input audio
        "-vn",              # Disable video stream if media container has one
        "-ac", "1",         # Mono channel
        "-ar", str(sample_rate), # 16,000 Hz
        "-c:a", "pcm_s16le",# 16-bit PCM
        output_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr}")
        raise AudioCleaningError(f"Failed to transcode audio file: {e.stderr[:200]}")

def normalize_gain(audio: np.ndarray, target_dbfs: float = -20.0, max_peak: float = 0.95) -> np.ndarray:
    """
    Normalizes audio gain/volume using RMS target with peak limiting
    to prevent clipping and standardize loudness across diverse recording conditions.
    """
    if len(audio) == 0:
        return audio
        
    rms = np.sqrt(np.mean(audio**2) + 1e-12)
    current_dbfs = 20 * np.log10(rms + 1e-12)
    gain_db = target_dbfs - current_dbfs
    gain_linear = 10 ** (gain_db / 20.0)
    
    normalized = audio * gain_linear
    peak = np.max(np.abs(normalized))
    if peak > max_peak:
        normalized = normalized * (max_peak / (peak + 1e-12))
        
    return normalized.astype(np.float32)

def strip_silence_vad(
    audio_int16: np.ndarray,
    sample_rate: int = 16000,
    frame_duration_ms: int = 30,
    aggressiveness: int = 2,
    hangover_frames: int = 5
) -> Tuple[np.ndarray, float, float]:
    """
    Applies webrtcvad on 30ms frames to strip non-speech silence gaps while
    preserving natural word pauses using a hangover smoothing window.
    
    Returns:
        voiced_samples: 1D np.ndarray of active speech samples (int16)
        voiced_duration_sec: total speech duration
        total_duration_sec: original audio duration
    """
    vad = webrtcvad.Vad(aggressiveness)
    frame_samples = int(sample_rate * frame_duration_ms / 1000)  # 480 samples @ 16kHz
    total_samples = len(audio_int16)
    total_duration_sec = total_samples / sample_rate
    
    if total_samples < frame_samples:
        return np.array([], dtype=np.int16), 0.0, total_duration_sec

    # Evaluate each frame
    num_frames = total_samples // frame_samples
    frame_decisions = []
    
    for i in range(num_frames):
        start = i * frame_samples
        end = start + frame_samples
        frame_bytes = audio_int16[start:end].tobytes()
        try:
            is_speech = vad.is_speech(frame_bytes, sample_rate)
        except Exception:
            # Fallback to energy check if VAD fails on corrupted buffer
            energy = np.mean(np.abs(audio_int16[start:end]))
            is_speech = energy > 300
        frame_decisions.append(is_speech)
        
    # Apply hangover smoothing algorithm:
    # If a speech frame is detected, keep subsequent 'hangover_frames' as speech
    # to avoid clipping word endings and micro-pauses.
    smoothed_decisions = list(frame_decisions)
    hangover_counter = 0
    for i in range(len(frame_decisions)):
        if frame_decisions[i]:
            hangover_counter = hangover_frames
        elif hangover_counter > 0:
            smoothed_decisions[i] = True
            hangover_counter -= 1

    # Extract all voiced frames
    voiced_chunks = []
    for i, is_speech in enumerate(smoothed_decisions):
        if is_speech:
            start = i * frame_samples
            end = start + frame_samples
            voiced_chunks.append(audio_int16[start:end])
            
    if voiced_chunks:
        voiced_audio = np.concatenate(voiced_chunks)
    else:
        voiced_audio = np.array([], dtype=np.int16)
        
    voiced_duration_sec = len(voiced_audio) / sample_rate
    return voiced_audio, voiced_duration_sec, total_duration_sec

def segment_into_uniform_chunks(
    audio_float: np.ndarray,
    sample_rate: int = 16000,
    chunk_duration_sec: float = 4.0
) -> List[np.ndarray]:
    """
    Chunks normalized continuous speech into uniform sub-windows of 4.0 seconds.
    If the trailing window is shorter than 4.0s but >= 1.5s, it is padded with
    reflective audio to maintain uniform window shape.
    """
    chunk_size = int(sample_rate * chunk_duration_sec)  # 64,000 samples
    total_samples = len(audio_float)
    
    if total_samples == 0:
        return []
        
    chunks = []
    for start in range(0, total_samples, chunk_size):
        end = start + chunk_size
        chunk = audio_float[start:end]
        
        if len(chunk) == chunk_size:
            chunks.append(chunk)
        elif len(chunk) >= int(sample_rate * 1.5):
            # Pad short final chunk with repeated/reflected segment to reach exact chunk_size
            pad_needed = chunk_size - len(chunk)
            padded = np.pad(chunk, (0, pad_needed), mode="reflect")
            chunks.append(padded)
        elif len(chunks) == 0:
            # If total audio is less than 1.5s, pad to 4s to allow inference attempt
            pad_needed = chunk_size - len(chunk)
            padded = np.pad(chunk, (0, pad_needed), mode="wrap")
            chunks.append(padded)
            
    return chunks

def process_raw_audio(
    file_path: str,
    temp_dir: Optional[str] = None
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Full Audio Cleaning Engine Pipeline:
    1. Converts raw input audio to standard 16 kHz mono WAV.
    2. Strips silence using webrtcvad with hangover smoothing.
    3. Normalizes gain/volume.
    4. Segments speech into uniform 4-second sub-windows.
    5. Returns chunks and telemetry dictionary.
    """
    target_dir = temp_dir or str(settings.TEMP_DIR)
    os.makedirs(target_dir, exist_ok=True)
    
    converted_wav_path = os.path.join(
        target_dir, f"transcoded_{os.path.basename(file_path)}.wav"
    )
    
    try:
        # Step 1: Transcode to 16 kHz Mono WAV
        convert_to_standard_wav(file_path, converted_wav_path, sample_rate=settings.SAMPLE_RATE)
        
        # Step 2: Read 16-bit PCM for VAD
        data_int16, sr = sf.read(converted_wav_path, dtype="int16")
        
        # Step 3: Silence stripping via VAD
        voiced_int16, voiced_dur, total_dur = strip_silence_vad(
            data_int16,
            sample_rate=settings.SAMPLE_RATE,
            frame_duration_ms=settings.VAD_FRAME_MS,
            aggressiveness=settings.VAD_AGGRESSIVENESS
        )
        
        # Check if voiced speech is adequate
        if voiced_dur < settings.MIN_VOICED_DURATION_SEC:
            # Check if original audio had non-silent energy before giving up
            data_float = data_int16.astype(np.float32) / 32768.0
            peak_energy = np.max(np.abs(data_float)) if len(data_float) > 0 else 0
            if peak_energy < 0.01 or len(data_float) < settings.SAMPLE_RATE * 0.5:
                raise InsufficientVoicedAudioError(
                    f"Voiced speech duration ({voiced_dur:.2f}s) is below threshold ({settings.MIN_VOICED_DURATION_SEC}s)."
                )
            # If audio has sound but VAD was overly aggressive, fall back to energy-gated frames
            voiced_float = data_float
            voiced_dur = total_dur
        else:
            # Convert int16 to float32 in [-1.0, 1.0]
            voiced_float = voiced_int16.astype(np.float32) / 32768.0

        # Step 4: Gain normalization
        normalized_speech = normalize_gain(voiced_float, target_dbfs=-20.0)
        
        # Step 5: Uniform 4-second chunking
        chunks = segment_into_uniform_chunks(
            normalized_speech,
            sample_rate=settings.SAMPLE_RATE,
            chunk_duration_sec=settings.CHUNK_DURATION_SEC
        )
        
        metadata = {
            "total_duration_sec": round(float(total_dur), 2),
            "voiced_duration_sec": round(float(voiced_dur), 2),
            "chunk_count": len(chunks),
            "sample_rate": settings.SAMPLE_RATE,
            "silence_ratio": round(float(max(0.0, 1.0 - (voiced_dur / (total_dur + 1e-6)))), 2)
        }
        
        return chunks, metadata
        
    finally:
        # Strictly purge temporary intermediate WAV file from volatile memory
        if os.path.exists(converted_wav_path):
            try:
                os.remove(converted_wav_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp WAV {converted_wav_path}: {e}")

def purge_file(file_path: Optional[str]) -> None:
    """
    Safely purges any temporary file from disk memory for privacy compliance.
    """
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Purged volatile audio file: {file_path}")
        except Exception as e:
            logger.warning(f"Error purging file {file_path}: {e}")
