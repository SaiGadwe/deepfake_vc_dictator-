from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime

class RationaleItem(BaseModel):
    feature: str = Field(..., description="Acoustic feature or metric name")
    description: str = Field(..., description="Human-readable explanation of the finding")
    direction: str = Field(..., description="'supports_ai', 'supports_human', or 'neutral'")
    impact_score: float = Field(default=0.0, description="Relative SHAP or deviation impact")

class ChunkScore(BaseModel):
    chunk_index: int
    start_time_sec: float
    end_time_sec: float
    score: float = Field(..., ge=0.0, le=1.0)

class SpeakerAnalysis(BaseModel):
    speaker_id: str = "speaker_0"
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Synthetic voice probability")
    verdict: str = Field(..., description="Must be 'Likely AI', 'Uncertain', 'Likely Human', or 'Can't Tell'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    chunk_count: int
    voiced_duration_sec: float = 0.0
    is_primary_threat: bool = False
    separation_mode: str = "mono_single"
    timestamps: List[dict] = []
    chunk_scores: List[ChunkScore] = []
    rationales: List[RationaleItem] = []

    @field_validator("verdict")
    @classmethod
    def validate_verdict(cls, v: str) -> str:
        allowed = {"Likely AI", "Uncertain", "Likely Human", "Can't Tell"}
        if v not in allowed:
            raise ValueError(f"Verdict '{v}' violates defensive messaging guardrails. Must be one of {allowed}")
        return v

class AnalysisResponse(BaseModel):
    analysis_id: str
    timestamp: datetime
    overall_verdict: str
    overall_risk_score: float
    speakers: List[SpeakerAnalysis]
    duration_analyzed_sec: float
    sample_rate: int = 16000
    processing_time_sec: float
    message: str

    @field_validator("overall_verdict")
    @classmethod
    def validate_verdict(cls, v: str) -> str:
        allowed = {"Likely AI", "Uncertain", "Likely Human", "Can't Tell"}
        if v not in allowed:
            raise ValueError(f"Verdict '{v}' violates defensive messaging guardrails. Must be one of {allowed}")
        return v

class FeedbackRequest(BaseModel):
    analysis_id: str
    feedback: str = Field(..., description="'accurate', 'inaccurate', or text note")

class FeedbackResponse(BaseModel):
    success: bool
    analysis_id: str
    feedback: str
    message: str

class AnalysisHistoryItem(BaseModel):
    analysis_id: str
    timestamp: datetime
    speaker_id: str
    risk_score: float
    verdict: str
    duration_analyzed: Optional[float] = None
    user_feedback: Optional[str] = None
