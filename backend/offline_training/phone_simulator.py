import os
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import soundfile as sf
import imageio_ffmpeg

logger = logging.getLogger(__name__)

def get_ffmpeg_exe() -> str:
    """Returns absolute path to the FFmpeg executable."""
    import shutil
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(f"FFmpeg binary not found: {e}")

class PhoneDegradationSimulator:
    """
    Offline Phone Network Degradation Simulator for ASVspoof 5 Augmentation.
    Applies realistic physical channel degradations encountered in telephone,
    cellular (AMR-NB / VoLTE), and VoIP (Opus / WebRTC) environments.
    """
    def __init__(self):
        self.ffmpeg_bin = get_ffmpeg_exe()

    def apply_pstn_bandpass(self, input_wav: str, output_wav: str, sample_rate: int = 16000) -> str:
        """
        Simulates traditional PSTN / Landline narrowband frequency response:
        G.711 300 Hz – 3,400 Hz bandpass filter, downsampled to 8 kHz, then resampled to 16 kHz.
        """
        cmd = [
            self.ffmpeg_bin, "-y",
            "-i", input_wav,
            "-af", "highpass=f=300,lowpass=f=3400",
            "-ar", "8000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            output_wav
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        # Resample back to standard 16 kHz WAV for uniform feature extraction
        resampled_wav = output_wav + "_16k.wav"
        cmd_resample = [
            self.ffmpeg_bin, "-y",
            "-i", output_wav,
            "-ar", str(sample_rate),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            resampled_wav
        ]
        subprocess.run(cmd_resample, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if os.path.exists(output_wav):
            os.remove(output_wav)
        os.rename(resampled_wav, output_wav)
        return output_wav

    def apply_amr_nb_codec(self, input_wav: str, output_wav: str, bitrate: str = "7.40k", sample_rate: int = 16000) -> str:
        """
        Simulates 2G/3G Cellular speech encoding via AMR-NB (Adaptive Multi-Rate Narrowband).
        Encodes to AMR @ 8 kHz, then decodes back to 16 kHz PCM WAV.
        """
        temp_amr = output_wav + ".amr"
        try:
            # Step 1: Encode to AMR-NB
            cmd_enc = [
                self.ffmpeg_bin, "-y",
                "-i", input_wav,
                "-ar", "8000",
                "-ac", "1",
                "-c:a", "libopencore_amrnb",
                "-b:a", bitrate,
                temp_amr
            ]
            subprocess.run(cmd_enc, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            # Step 2: Decode AMR-NB to 16 kHz WAV
            cmd_dec = [
                self.ffmpeg_bin, "-y",
                "-i", temp_amr,
                "-ar", str(sample_rate),
                "-ac", "1",
                "-c:a", "pcm_s16le",
                output_wav
            ]
            subprocess.run(cmd_dec, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return output_wav
        finally:
            if os.path.exists(temp_amr):
                os.remove(temp_amr)

    def apply_opus_voip_codec(self, input_wav: str, output_wav: str, bitrate: str = "12k", sample_rate: int = 16000) -> str:
        """
        Simulates modern VoIP / WebRTC / WhatsApp call compression via the Opus codec
        at constrained mobile network bitrates (e.g., 12–16 kbps).
        """
        temp_opus = output_wav + ".ogg"
        try:
            # Step 1: Compress to Opus
            cmd_enc = [
                self.ffmpeg_bin, "-y",
                "-i", input_wav,
                "-c:a", "libopus",
                "-b:a", bitrate,
                temp_opus
            ]
            subprocess.run(cmd_enc, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            # Step 2: Decompress to 16 kHz PCM WAV
            cmd_dec = [
                self.ffmpeg_bin, "-y",
                "-i", temp_opus,
                "-ar", str(sample_rate),
                "-ac", "1",
                "-c:a", "pcm_s16le",
                output_wav
            ]
            subprocess.run(cmd_dec, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return output_wav
        finally:
            if os.path.exists(temp_opus):
                os.remove(temp_opus)

    def apply_packet_loss_jitter(self, audio_data: np.ndarray, loss_rate: float = 0.03, sample_rate: int = 16000) -> np.ndarray:
        """
        Simulates VoIP packet dropouts and packet loss concealment (PLC) artifacts.
        Randomly zeroes or attenuates 20–30 ms frames.
        """
        frame_len = int(sample_rate * 0.025)  # 25 ms frame (400 samples)
        total_samples = len(audio_data)
        num_frames = total_samples // frame_len
        
        degraded = np.copy(audio_data)
        for f in range(num_frames):
            if np.random.rand() < loss_rate:
                start = f * frame_len
                end = start + frame_len
                # Zero out with smooth edge fade to prevent loud clicks
                fade = np.linspace(1.0, 0.0, 30)
                degraded[start:start+30] *= fade
                degraded[start+30:end-30] = 0.0
                degraded[end-30:end] *= fade[::-1]
                
        return degraded

    def augment_audio_file(
        self,
        input_path: str,
        output_path: str,
        degradation_type: str = "random" # "pstn" | "amr_nb" | "opus" | "packet_loss" | "random"
    ) -> str:
        """
        Applies a selected or randomly sampled telephone network degradation.
        """
        if degradation_type == "random":
            degradation_type = np.random.choice(["pstn", "amr_nb", "opus", "packet_loss"])

        if degradation_type == "pstn":
            return self.apply_pstn_bandpass(input_path, output_path)
        elif degradation_type == "amr_nb":
            return self.apply_amr_nb_codec(input_path, output_path, bitrate="7.40k")
        elif degradation_type == "opus":
            return self.apply_opus_voip_codec(input_path, output_path, bitrate="12k")
        elif degradation_type == "packet_loss":
            # Load audio, apply packet loss, write to output
            data, sr = sf.read(input_path, dtype="float32")
            if data.ndim > 1:
                data = data[:, 0]
            lossy = self.apply_packet_loss_jitter(data, loss_rate=0.04, sample_rate=sr)
            sf.write(output_path, lossy, sr)
            return output_path
        else:
            return self.apply_pstn_bandpass(input_path, output_path)
