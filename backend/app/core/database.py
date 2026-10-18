from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.core.config import settings

# SQLite connection with thread check disabled for FastAPI async workers
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """
    FastAPI dependency yielding a managed database session.
    Guarantees session closure after request lifecycle.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Ensure schema tables exist
Base.metadata.create_all(bind=engine)

