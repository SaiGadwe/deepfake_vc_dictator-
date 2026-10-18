# AegisVoice: Voice Authenticity & Synthetic Voice Detection

An end-to-end Voice Authenticity and Synthetic Voice Detection system featuring an ephemeral audio processing pipeline, multi-speaker diarization, deterministic defensive verdicts, explainability via SHAP, and an ASVspoof 5 phone network simulation offline training suite.

---

## 🏛 System Architecture & Phased Design

```
┌────────────────────────────────────────────────────────┐
│               Upload Screen (React PWA)                │
│       File Picker, Live Mic Recorder, Wave Preview     │
└───────────────────────────┬────────────────────────────┘
                            │ POST /api/v1/analyze
                            ▼
┌────────────────────────────────────────────────────────┐
│                   API Server (FastAPI)                 │
│         Volatile temp buffering, Async pipeline        │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│                  Audio Cleaning Engine                 │
│   • FFmpeg transcoding to 16 kHz mono WAV              │
│   • Gain normalization (-20 dBFS target, peak limit)   │
│   • webrtcvad silence gap stripping + hangover filter  │
│   • Uniform 4-second sub-window chunking               │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│         Voice Splitting (Phase 2 Additions)            │
│   • Dual-channel stereo separation (Channel L & R)     │
│   • Mixed single-channel KMeans acoustic diarization   │
│   • Per-speaker sub-window chunk & timestamp routing   │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│           Feature Extraction (201-dim vector)          │
│   • MFCCs (1-40) + Mean, Variance, Skew (120 features) │
│   • LFCCs (1-20) via linear filterbank (60 features)   │
│   • Praat Pitch / F0, Jitter & Shimmer (Parselmouth)   │
│   • Spectral Centroid, Rolloff, Flatness & RMS Energy  │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│                   ML Inference Engine                  │
│   • Tuned XGBoost classifier (P(fake) per 4s chunk)    │
│   • Chunk Score Aggregation (Weighted Mean + Anomaly)  │
│   • SHAP TreeExplainer rationale generation            │
└───────────────┬────────────────────────────┬───────────┘
                ▼                            ▼
┌───────────────────────────────┐ ┌──────────────────────┐
│         Verdict Logic         │ │   Results Database   │
│ Strictly mapped to:           │ │   SQLite metadata    │
│ "Likely AI" | "Uncertain"     │ │   Scores only        │
│ "Likely Human" | "Can't Tell" │ │   NO raw audio!      │
│ (Forbidden: "Safe"/"Genuine") │ └──────────────────────┘
└───────────────┬───────────────┘
                ▼
┌────────────────────────────────────────────────────────┐
│                Result Screen (React PWA)               │
│   • Suspicious score circular SVG progress ring        │
│   • Guarded verdict badge & defensive explanation      │
│   • Multi-voice card layout (Top threat prioritized)   │
│   • One-touch playback per separated voice chunk       │
│   • Key SHAP explainability indicators                 │
│   • User accuracy feedback loop                        │
└────────────────────────────────────────────────────────┘
```

---

## 🔒 Architectural Rules & Guardrails
- **Privacy First**: Audio files are held exclusively in ephemeral volatile memory, processed, and purged in a `finally` block immediately post-inference. Only risk scores, voice segment metadata, and user feedback are stored in SQLite.
- **Defensive Messaging**: The verdict logic strictly prevents false guarantees and forbids words like `"Safe"` or `"Genuine"`. All evaluations map strictly to `"Likely AI"`, `"Uncertain"`, `"Likely Human"`, or `"Can't Tell"`.

---

## 🚀 Quickstart

### 1. Backend Server
```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- Swagger API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Interactive Health Check: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

### 2. Mobile PWA Frontend
```bash
cd frontend
npm run dev
```
Open in browser / mobile simulator: `http://localhost:5173`

### 3. Run Pipeline Integration Tests
```bash
cd backend
python tests/test_pipeline.py
```

### 4. Run Offline ASVspoof 5 & Phone Simulation Training Pipeline
```bash
cd backend
python offline_training/train_and_evaluate.py --samples 40
```
This runs:
- Phone network degradation simulation: PSTN G.711 (300-3400 Hz) bandpass, 8 kHz downsampling, 2G/3G AMR-NB cellular codec compression, and low-bitrate Opus VoIP compression.
- 201-dim MFCC + LFCC + Parselmouth F0 feature extraction across degraded datasets.
- XGBoost classifier training, hyperparameter tuning, and Equal Error Rate (EER) / ROC-AUC calculation.
- Direct export of optimized weights into `backend/models_storage/xgboost_voice_model.joblib`.
- Ephemeral audio purging post-training.
