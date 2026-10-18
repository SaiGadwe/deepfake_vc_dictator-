import datetime
from sqlalchemy import Column, String, Float, DateTime, Text, Integer
from app.core.database import Base

class AnalysisRecord(Base):
    """
    SQLAlchemy ORM Model storing inference audit metadata and risk scores.
    STRICT PRIVACY GUARDRAIL:
    Under NO circumstances should raw audio bytes or persistent audio file paths
    be persisted in this schema.
    """
    __tablename__ = "analysis_records"

    # Unique identifier for the transaction
    analysis_id = Column(String(64), primary_key=True, index=True)
    
    # Audit timestamp
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    
    # Speaker identification tag (e.g. "speaker_0", "voice_1")
    speaker_id = Column(String(32), default="speaker_0", nullable=False)
    
    # Calculated synthetic voice risk score [0.0 - 1.0]
    risk_score = Column(Float, nullable=False)
    
    # Confidence or model uncertainty measure [0.0 - 1.0]
    confidence = Column(Float, nullable=False, default=0.5)
    
    # Strictly defensive verdict: "Likely AI" | "Uncertain" | "Likely Human" | "Can't Tell"
    # Note: Words "Safe" and "Genuine" are strictly forbidden.
    verdict = Column(String(32), nullable=False)
    
    # Explainability rationale and acoustic indicators (JSON string)
    rationale_json = Column(Text, nullable=True)
    
    # Processed speech metadata
    duration_analyzed = Column(Float, nullable=True)
    chunk_count = Column(Integer, default=1)
    
    # User feedback loop ("accurate", "inaccurate", or comments)
    user_feedback = Column(String(255), nullable=True)

    def __repr__(self):
        return f"<AnalysisRecord(id={self.analysis_id}, verdict={self.verdict}, score={self.risk_score:.2f})>"

from app.core.database import engine
Base.metadata.create_all(bind=engine)

