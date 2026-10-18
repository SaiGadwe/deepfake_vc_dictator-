import os
from pathlib import Path
from typing import Set

class Settings:
    PROJECT_NAME: str = "Voice Authenticity & Synthetic Voice Detection"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Audio preprocessing specifications
    SAMPLE_RATE: int = 16000  # 16 kHz mono
    CHUNK_DURATION_SEC: float = 4.0
    CHUNK_SAMPLES: int = int(SAMPLE_RATE * CHUNK_DURATION_SEC)  # 64,000 samples
    MIN_VOICED_DURATION_SEC: float = 1.0  # Minimum voiced audio required for confident inference
    
    # VAD Configuration (webrtcvad: 10, 20, or 30 ms frame duration)
    VAD_FRAME_MS: int = 30
    VAD_AGGRESSIVENESS: int = 2  # 0: least aggressive, 3: most aggressive
    
    # Database
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'voice_authenticity.db'}")
    
    # Storage & Privacy: Strictly ephemeral storage for inference
    TEMP_DIR: Path = BASE_DIR / "temp_audio"
    MODELS_DIR: Path = BASE_DIR / "models_storage"
    DEFAULT_MODEL_PATH: Path = MODELS_DIR / "xgboost_voice_model.joblib"
    
    # Verdict Guardrails & Thresholds
    # Strictly prohibited words: "Safe", "Genuine"
    FORBIDDEN_VERDICTS: Set[str] = {"Safe", "Genuine", "Real", "Original"}
    
    # Valid deterministic verdict outputs
    VERDICT_LIKELY_AI: str = "Likely AI"
    VERDICT_UNCERTAIN: str = "Uncertain"
    VERDICT_LIKELY_HUMAN: str = "Likely Human"
    VERDICT_CANT_TELL: str = "Can't Tell"
    
    # Score thresholds for risk probability P(fake) in [0.0, 1.0]
    THRESHOLD_AI: float = 0.65       # >= 0.65 -> Likely AI
    THRESHOLD_HUMAN: float = 0.35    # < 0.35 -> Likely Human
    # Between 0.35 and 0.65 -> Uncertain
    
    # Allowed Audio Extensions
    ALLOWED_EXTENSIONS: Set[str] = {".wav", ".mp3", ".ogg", ".m4a", ".aac", ".flac", ".wma", ".webm"}

settings = Settings()
settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
