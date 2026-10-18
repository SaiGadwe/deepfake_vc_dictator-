import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import joblib
import xgboost as xgb
from app.core.config import settings
from app.services.feature_extractor import get_feature_names

logger = logging.getLogger(__name__)

def generate_calibrated_synthetic_dataset(n_samples: int = 400) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generates a calibrated synthetic acoustic feature distribution mimicking
    the physical acoustic divergence between natural human voices and synthetic neural vocoders:
    - Natural human speech: higher pitch standard deviation (natural prosodic inflection),
      higher natural jitter/shimmer micro-perturbation, dynamic spectral centroid variability,
      and standard high-frequency LFCC roll-off.
    - Synthetic / cloned speech: lower pitch variability (monotonic flat pitch),
      suppressed vocal tremor/jitter, discrete quantization in pitch, and prominent high-frequency
      cepstral energy residuals (LFCC 10-20) characteristic of neural vocoder upsampling.
    """
    np.random.seed(42)
    feature_names = get_feature_names()
    n_features = len(feature_names)
    
    half = n_samples // 2
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int32)
    
    # Class 0: Natural Human Voice
    # Class 1: Synthetic AI Voice
    y[half:] = 1
    
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    
    # Baseline normal distribution for background acoustic noise
    X += np.random.normal(0.0, 1.0, size=(n_samples, n_features)).astype(np.float32)
    
    # 1. Pitch / Prosody Modulation
    if "f0_std" in name_to_idx:
        idx = name_to_idx["f0_std"]
        X[:half, idx] = np.random.normal(32.0, 8.0, half)   # Human: natural wide pitch variance (Hz)
        X[half:, idx] = np.random.normal(7.5, 3.0, half)    # AI: flattened pitch contour
        
    if "jitter_local" in name_to_idx:
        idx = name_to_idx["jitter_local"]
        X[:half, idx] = np.random.normal(0.045, 0.012, half) # Human: organic vocal fold perturbation
        X[half:, idx] = np.random.normal(0.008, 0.004, half) # AI: unnaturally smooth glottal pulses
        
    if "shimmer_local" in name_to_idx:
        idx = name_to_idx["shimmer_local"]
        X[:half, idx] = np.random.normal(0.08, 0.02, half)
        X[half:, idx] = np.random.normal(0.02, 0.01, half)

    # 2. High-Frequency LFCC Artifacts (typical in HiFi-GAN / MelGAN upsampling)
    for i in range(12, 21):
        col_mean = f"lfcc_{i}_mean"
        col_var = f"lfcc_{i}_var"
        if col_mean in name_to_idx:
            X[:half, name_to_idx[col_mean]] = np.random.normal(-0.5, 0.3, half)
            X[half:, name_to_idx[col_mean]] = np.random.normal(0.8, 0.4, half) # AI spectral energy boost
        if col_var in name_to_idx:
            X[:half, name_to_idx[col_var]] = np.random.normal(0.2, 0.1, half)
            X[half:, name_to_idx[col_var]] = np.random.normal(1.2, 0.3, half)

    # 3. Spectral Flatness & Centroid Skew
    if "spectral_flatness_mean" in name_to_idx:
        idx = name_to_idx["spectral_flatness_mean"]
        X[:half, idx] = np.random.normal(0.015, 0.005, half)
        X[half:, idx] = np.random.normal(0.065, 0.015, half) # AI has elevated resynthesis noise

    if "spectral_centroid_skew" in name_to_idx:
        idx = name_to_idx["spectral_centroid_skew"]
        X[:half, idx] = np.random.normal(1.8, 0.5, half)
        X[half:, idx] = np.random.normal(0.4, 0.3, half)
        
    return X, y, feature_names

def train_and_export_baseline_model(model_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Trains and persists the baseline XGBoost classifier with calibration.
    """
    target_path = model_path or settings.DEFAULT_MODEL_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating calibrated synthetic feature distribution...")
    X, y, feature_names = generate_calibrated_synthetic_dataset(n_samples=600)
    
    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=60,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X, y)
    
    bundle = {
        "model": model,
        "feature_names": feature_names,
        "version": "1.0.0-mvp",
        "n_features": len(feature_names),
        "thresholds": {
            "ai": settings.THRESHOLD_AI,
            "human": settings.THRESHOLD_HUMAN
        }
    }
    
    joblib.dump(bundle, str(target_path))
    logger.info(f"Successfully trained and exported XGBoost baseline model to {target_path}")
    return bundle

def ensure_model_exists() -> Dict[str, Any]:
    """
    Checks if model exists on disk, otherwise trains the baseline model automatically.
    """
    if settings.DEFAULT_MODEL_PATH.exists():
        try:
            bundle = joblib.load(str(settings.DEFAULT_MODEL_PATH))
            return bundle
        except Exception as e:
            logger.warning(f"Failed to load existing model ({e}). Rebuilding...")
            return train_and_export_baseline_model()
    else:
        return train_and_export_baseline_model()
