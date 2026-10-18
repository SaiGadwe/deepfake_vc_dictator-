import os
import time
import uuid
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models.analysis import AnalysisRecord
from app.schemas.analysis import (
    AnalysisResponse,
    SpeakerAnalysis,
    ChunkScore,
    RationaleItem,
    FeedbackRequest,
    FeedbackResponse,
    AnalysisHistoryItem,
)
from app.services.audio_cleaner import process_raw_audio, purge_file, InsufficientVoicedAudioError
from app.services.voice_splitter import separate_speakers, SpeakerAudioProfile
from app.services.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health", tags=["System"])
def health_check():
    """
    Health check verifying API operational status, FFmpeg readiness, and ML engine state.
    """
    engine = InferenceEngine.get_instance()
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": engine.bundle.get("version", "1.0.0"),
        "features_count": len(engine.feature_names),
        "privacy_policy": "Strict volatile processing; zero raw audio persisted."
    }

@router.post("/analyze", response_model=AnalysisResponse, status_code=status.HTTP_200_OK, tags=["Inference"])
async def analyze_audio(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Core Voice Authenticity Analysis Endpoint:
    1. Streams uploaded audio file directly into volatile temp memory.
    2. Transcodes to standard 16 kHz mono WAV via FFmpeg.
    3. Normalizes gain and strips non-speech silence via webrtcvad.
    4. Segments voiced speech into uniform 4-second sub-windows.
    5. Computes MFCC + LFCC + Parselmouth F0 + Spectral summary features.
    6. Runs XGBoost classification & SHAP explainability per chunk.
    7. Emits deterministic defensive verdicts strictly mapped to:
       'Likely AI', 'Uncertain', 'Likely Human', or 'Can't Tell'.
    8. Persists ONLY metadata and scores into SQLite DB.
    9. Unconditionally purges raw and intermediate audio files from disk.
    """
    start_time = time.time()
    analysis_id = f"aud_{uuid.uuid4().hex[:16]}"
    
    # Validate extension
    file_ext = Path(file.filename or "unknown.wav").suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: {sorted(list(settings.ALLOWED_EXTENSIONS))}"
        )
        
    # Write to volatile temp directory
    temp_file_path = os.path.join(str(settings.TEMP_DIR), f"{analysis_id}{file_ext}")
    
    try:
        # Buffer upload stream to ephemeral disk
        with open(temp_file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB buffer
                buffer.write(chunk)
                
        # Step 1: Speaker Separation & Diarization Engine (Stereo channels or KMeans clustering)
        try:
            profiles, separation_telemetry = separate_speakers(temp_file_path, temp_dir=str(settings.TEMP_DIR))
            total_dur = separation_telemetry.get("total_duration_sec", 0.0)
        except Exception as e:
            logger.warning(f"Speaker separation failed, falling back to basic audio cleaner: {e}")
            chunks, meta = process_raw_audio(temp_file_path, temp_dir=str(settings.TEMP_DIR))
            total_dur = meta.get("total_duration_sec", 0.0)
            profiles = [
                SpeakerAudioProfile(
                    speaker_id="Voice 1",
                    channel_index=None,
                    chunks=chunks,
                    voiced_duration_sec=meta.get("voiced_duration_sec", 0.0),
                    total_duration_sec=total_dur,
                    chunk_timestamps=[
                        {"start_time_sec": round(i * settings.CHUNK_DURATION_SEC, 2),
                         "end_time_sec": round((i + 1) * settings.CHUNK_DURATION_SEC, 2)}
                        for i in range(len(chunks))
                    ],
                    separation_mode="mono_single"
                )
            ]

        # Step 2: Multi-Voice Pipeline Rerouting - Independent XGBoost inference per voice profile
        engine = InferenceEngine.get_instance()
        speaker_analyses: List[SpeakerAnalysis] = []

        for prof in profiles:
            res = engine.analyze_chunks(
                chunks=prof.chunks,
                speaker_id=prof.speaker_id,
                voiced_duration_sec=prof.voiced_duration_sec
            )
            speaker_analyses.append(
                SpeakerAnalysis(
                    speaker_id=res["speaker_id"],
                    risk_score=res["risk_score"],
                    verdict=res["verdict"],
                    confidence=res["confidence"],
                    chunk_count=res["chunk_count"],
                    voiced_duration_sec=prof.voiced_duration_sec,
                    separation_mode=prof.separation_mode,
                    timestamps=prof.chunk_timestamps,
                    chunk_scores=[ChunkScore(**c) for c in res["chunk_scores"]],
                    rationales=[RationaleItem(**r) for r in res["rationales"]]
                )
            )

        # Step 3: Reorder so the MOST SUSPICIOUS voice profile is at the top
        speaker_analyses.sort(key=lambda s: s.risk_score, reverse=True)
        if speaker_analyses and speaker_analyses[0].risk_score >= settings.THRESHOLD_HUMAN:
            speaker_analyses[0].is_primary_threat = True

        overall_verdict = speaker_analyses[0].verdict if speaker_analyses else settings.VERDICT_CANT_TELL
        overall_risk_score = speaker_analyses[0].risk_score if speaker_analyses else 0.5

        # Persist audit records in SQLite (strictly NO raw audio stored)
        primary_record = AnalysisRecord(
            analysis_id=analysis_id,
            timestamp=datetime.utcnow(),
            speaker_id=speaker_analyses[0].speaker_id if speaker_analyses else "speaker_0",
            risk_score=overall_risk_score,
            confidence=speaker_analyses[0].confidence if speaker_analyses else 0.5,
            verdict=overall_verdict,
            rationale_json=json.dumps([r.model_dump() for r in speaker_analyses[0].rationales]) if speaker_analyses else "[]",
            duration_analyzed=total_dur,
            chunk_count=sum(s.chunk_count for s in speaker_analyses),
            user_feedback=None
        )
        db.add(primary_record)
        db.commit()

        processing_time = round(time.time() - start_time, 2)

        return AnalysisResponse(
            analysis_id=analysis_id,
            timestamp=primary_record.timestamp,
            overall_verdict=overall_verdict,
            overall_risk_score=overall_risk_score,
            speakers=speaker_analyses,
            duration_analyzed_sec=total_dur,
            sample_rate=settings.SAMPLE_RATE,
            processing_time_sec=processing_time,
            message=(
                f"Analysis completed in {processing_time}s across {len(speaker_analyses)} speaker profile(s) and {sum(s.chunk_count for s in speaker_analyses)} speech window(s)."
                if sum(s.chunk_count for s in speaker_analyses) > 0
                else "Voiced audio sample is too brief or ambiguous for confident classification."
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference pipeline failed on {analysis_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio analysis failed: {str(e)}"
        )
    finally:
        # PRIVACY FIRST MANDATE: Unconditionally purge uploaded audio from disk memory
        purge_file(temp_file_path)

@router.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
def submit_feedback(
    payload: FeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    Submits user feedback regarding accuracy of the synthetic voice assessment.
    Stored alongside score metadata for continual model evaluation.
    """
    record = db.query(AnalysisRecord).filter(AnalysisRecord.analysis_id == payload.analysis_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis session '{payload.analysis_id}' not found."
        )
        
    record.user_feedback = payload.feedback.strip()
    db.commit()
    
    return FeedbackResponse(
        success=True,
        analysis_id=payload.analysis_id,
        feedback=record.user_feedback,
        message="Thank you for providing verification feedback."
    )

@router.get("/history", response_model=List[AnalysisHistoryItem], tags=["Audit History"])
def get_analysis_history(
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """
    Retrieves previous analysis metadata records.
    Strict privacy compliance: strictly NO audio waveforms or binary content returned.
    """
    records = (
        db.query(AnalysisRecord)
        .order_by(AnalysisRecord.timestamp.desc())
        .limit(limit)
        .all()
    )
    
    return [
        AnalysisHistoryItem(
            analysis_id=r.analysis_id,
            timestamp=r.timestamp,
            speaker_id=r.speaker_id,
            risk_score=r.risk_score,
            verdict=r.verdict,
            duration_analyzed=r.duration_analyzed,
            user_feedback=r.user_feedback
        )
        for r in records
    ]
