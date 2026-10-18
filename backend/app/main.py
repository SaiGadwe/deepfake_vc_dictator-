import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.database import Base, engine
from app.api.v1.router import router as api_v1_router
from app.services.inference_engine import InferenceEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Database tables
    logger.info("Initializing SQLite database tables...")
    Base.metadata.create_all(bind=engine)
    
    # Warmup ML inference engine & SHAP explainer
    logger.info("Warming up ML inference model and SHAP tree explainer...")
    InferenceEngine.get_instance()
    
    logger.info(f"{settings.PROJECT_NAME} v{settings.VERSION} ready to accept requests.")
    yield
    logger.info("Shutting down Voice Authenticity backend.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="End-to-End Voice Authenticity & Synthetic Voice Detection API Engine",
    lifespan=lifespan
)

# Enable CORS for Mobile Apps (React Native, Flutter, Expo) and Web PWAs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 router
app.include_router(api_v1_router, prefix=settings.API_V1_STR)

@app.get("/", tags=["Root"])
def root_status():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "api_v1": f"{settings.API_V1_STR}/analyze",
        "privacy_mandate": "Raw audio is strictly ephemeral and purged immediately post-inference."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
