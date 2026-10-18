import os
import sys
import io
import time
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

# Ensure backend root is in python path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.main import app
from app.core.config import settings
from app.core.database import Base, engine

client = TestClient(app)

def create_in_memory_wav(duration_sec=6.0, freq=220.0, sample_rate=16000) -> io.BytesIO:
    """Creates a temporary in-memory 16kHz WAV buffer simulating speech with pause."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Pitch modulated tone simulating natural human inflection
    f_mod = freq + 15.0 * np.sin(2 * np.pi * 3.0 * t)
    phase = 2 * np.pi * np.cumsum(f_mod) / sample_rate
    audio = 0.4 * np.sin(phase) + 0.05 * np.random.randn(len(t))
    # Add a 0.5s pause in between
    pause_start = int(sample_rate * 2.0)
    pause_end = int(sample_rate * 2.5)
    audio[pause_start:pause_end] = 0.0
    
    bio = io.BytesIO()
    sf.write(bio, audio.astype(np.float32), sample_rate, format="WAV", subtype="PCM_16")
    bio.seek(0)
    return bio

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "privacy_policy" in data

def test_analyze_audio_pipeline():
    wav_buf = create_in_memory_wav(duration_sec=8.0)
    
    files = {
        "file": ("test_speech_sample.wav", wav_buf, "audio/wav")
    }
    
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 200, f"Error: {response.text}"
    data = response.json()
    
    # Assert defensive messaging guardrail
    verdict = data["overall_verdict"]
    assert verdict in ["Likely AI", "Uncertain", "Likely Human", "Can't Tell"], f"Invalid verdict: {verdict}"
    assert "Safe" not in verdict, "VIOLATION: Output contains forbidden word 'Safe'"
    assert "Genuine" not in verdict, "VIOLATION: Output contains forbidden word 'Genuine'"
    
    # Assert score bounds
    risk_score = data["overall_risk_score"]
    assert 0.0 <= risk_score <= 1.0
    
    # Assert speaker analysis
    assert len(data["speakers"]) >= 1
    speaker = data["speakers"][0]
    assert speaker["chunk_count"] >= 1
    assert len(speaker["rationales"]) > 0
    
    # Check rationale structure
    for rat in speaker["rationales"]:
        assert "feature" in rat
        assert "description" in rat
        assert rat["direction"] in ["supports_ai", "supports_human", "neutral"]
        
    analysis_id = data["analysis_id"]
    
    # Test feedback loop
    fb_res = client.post("/api/v1/feedback", json={
        "analysis_id": analysis_id,
        "feedback": "accurate"
    })
    assert fb_res.status_code == 200
    assert fb_res.json()["success"] is True

    # Test audit history endpoint
    hist_res = client.get("/api/v1/history?limit=5")
    assert hist_res.status_code == 200
    history = hist_res.json()
    assert len(history) >= 1
    assert any(h["analysis_id"] == analysis_id for h in history)
    
    # Verify no raw audio file remained in temp folder
    temp_files = list(settings.TEMP_DIR.glob(f"*{analysis_id}*"))
    assert len(temp_files) == 0, f"Privacy violation: Temporary files not purged: {temp_files}"
    print("PASS: Single voice analysis test passed.")

def test_stereo_call_separation():
    """Tests dual-channel stereo call recording separation."""
    sr = 16000
    duration = 8.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Left channel: Human-like modulated voice
    ch0 = 0.4 * np.sin(2 * np.pi * 140 * t) + 0.05 * np.random.randn(len(t))
    # Right channel: Higher frequency voice
    ch1 = 0.4 * np.sin(2 * np.pi * 320 * t) + 0.05 * np.random.randn(len(t))
    
    stereo_data = np.stack([ch0, ch1], axis=-1).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, stereo_data, sr, format="WAV", subtype="PCM_16")
    bio.seek(0)
    
    files = {"file": ("stereo_call_recording.wav", bio, "audio/wav")}
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 200, response.text
    data = response.json()
    
    assert len(data["speakers"]) == 2, f"Expected 2 stereo channels, got {len(data['speakers'])}"
    assert data["speakers"][0]["separation_mode"] == "stereo_channels"
    # Ensure most suspicious is at top
    assert data["speakers"][0]["risk_score"] >= data["speakers"][1]["risk_score"]
    # Check playback timestamps
    assert len(data["speakers"][0]["timestamps"]) > 0
    print("PASS: Stereo call recording separation test passed.")

def test_mono_kmeans_clustering():
    """Tests mixed mono call diarization via KMeans clustering."""
    sr = 16000
    # Speaker A: 8 seconds (140 Hz base with envelope)
    t1 = np.linspace(0, 8.0, int(sr * 8.0), endpoint=False)
    env1 = np.maximum(0.2, np.sin(2 * np.pi * 3.0 * t1))
    harm1 = 0.4 * np.sin(2 * np.pi * 140 * t1) + 0.3 * np.sin(2 * np.pi * 280 * t1)
    speech_A = (env1 * harm1 * 0.5).astype(np.float32)

    # Speaker B: 8 seconds (280 Hz base with envelope)
    t2 = np.linspace(0, 8.0, int(sr * 8.0), endpoint=False)
    env2 = np.maximum(0.2, np.sin(2 * np.pi * 4.0 * t2))
    harm2 = 0.4 * np.sin(2 * np.pi * 280 * t2) + 0.3 * np.sin(2 * np.pi * 560 * t2)
    speech_B = (env2 * harm2 * 0.5).astype(np.float32)

    mono_audio = np.concatenate([speech_A, speech_B])
    bio = io.BytesIO()
    sf.write(bio, mono_audio, sr, format="WAV", subtype="PCM_16")
    bio.seek(0)

    files = {"file": ("mono_mixed_call.wav", bio, "audio/wav")}
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 200, response.text
    data = response.json()

    assert len(data["speakers"]) == 2, f"Expected 2 clustered speakers, got {len(data['speakers'])}"
    assert data["speakers"][0]["separation_mode"] == "mono_clustered"
    assert data["speakers"][0]["risk_score"] >= data["speakers"][1]["risk_score"]
    print("PASS: Mono KMeans diarization clustering test passed.")

if __name__ == "__main__":
    test_health_check()
    test_analyze_audio_pipeline()
    test_stereo_call_separation()
    test_mono_kmeans_clustering()
    print("\nALL PHASE 1 & PHASE 2 TESTS PASSED WITH PRIVACY & GUARDRAIL COMPLIANCE!")

