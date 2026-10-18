import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import shap
from app.core.config import settings
from app.services.model_builder import ensure_model_exists
from app.services.feature_extractor import extract_chunk_features

logger = logging.getLogger(__name__)

# Feature explanation dictionary mapping acoustic metrics to layperson insights
FEATURE_EXPLANATIONS = {
    "f0_std": {
        "title": "Pitch & Prosodic Modulation",
        "ai_desc": "Abnormally flat and monotonic pitch contour characteristic of text-to-speech vocoders.",
        "human_desc": "Organic pitch dynamics and natural conversational inflection."
    },
    "jitter_local": {
        "title": "Vocal Fold Micro-Perturbation (Jitter)",
        "ai_desc": "Unnaturally smooth glottal pulse timing with absent vocal cord micro-tremor.",
        "human_desc": "Natural acoustic perturbation and biological vocal tract micro-variations."
    },
    "shimmer_local": {
        "title": "Glottal Amplitude Perturbation (Shimmer)",
        "ai_desc": "Synthetic amplitude stability failing to replicate physiological breath regulation.",
        "human_desc": "Natural cycle-to-cycle amplitude fluctuations."
    },
    "spectral_flatness_mean": {
        "title": "Resynthesis Noise & Spectral Flatness",
        "ai_desc": "High-frequency diffuse noise signature typical of neural diffusion or GAN vocoders.",
        "human_desc": "Clean harmonic harmonic-to-noise ratio across voice formants."
    },
    "spectral_centroid_skew": {
        "title": "Spectral Energy Distribution",
        "ai_desc": "Unusual harmonic distribution indicative of neural audio upsampling.",
        "human_desc": "Balanced biological formant resonance."
    }
}

class InferenceEngine:
    _instance = None

    def __init__(self):
        self.bundle = ensure_model_exists()
        self.model = self.bundle["model"]
        self.feature_names = self.bundle["feature_names"]
        self.explainer = shap.TreeExplainer(self.model)
        logger.info("InferenceEngine initialized with calibrated model and SHAP TreeExplainer.")

    @classmethod
    def get_instance(cls) -> "InferenceEngine":
        if cls._instance is None:
            cls._instance = InferenceEngine()
        return cls._instance

    def compute_verdict(self, risk_score: float, chunk_count: int, voiced_dur: float) -> str:
        """
        Deterministic verdict logic strictly bound to defensive messaging guardrails:
        Output MUST be one of:
        - 'Likely AI'
        - 'Uncertain'
        - 'Likely Human'
        - 'Can't Tell'
        
        CRITICAL GUARDRAIL: Prohibits 'Safe' or 'Genuine' from ever being emitted.
        """
        # Edge case: Insufficient speech or extreme brevity
        if chunk_count == 0 or voiced_dur < settings.MIN_VOICED_DURATION_SEC:
            verdict = settings.VERDICT_CANT_TELL
        elif risk_score >= settings.THRESHOLD_AI:
            verdict = settings.VERDICT_LIKELY_AI
        elif risk_score < settings.THRESHOLD_HUMAN:
            verdict = settings.VERDICT_LIKELY_HUMAN
        else:
            verdict = settings.VERDICT_UNCERTAIN

        # Defensive assertion guardrail
        if verdict in settings.FORBIDDEN_VERDICTS or "Safe" in verdict or "Genuine" in verdict:
            raise ValueError(f"Architectural guardrail violation: Forbidden verdict string '{verdict}' detected!")

        return verdict

    def generate_shap_rationales(self, feature_matrix: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Extracts top contributing SHAP features explaining the synthetic voice determination.
        """
        try:
            mean_features = np.mean(feature_matrix, axis=0, keepdims=True)
            shap_values = self.explainer.shap_values(mean_features)
            
            # Handle binary classifier output shape from SHAP
            if isinstance(shap_values, list) and len(shap_values) == 2:
                # Class 1 (AI) shap values
                sv = shap_values[1][0]
            elif len(shap_values.shape) == 3:
                sv = shap_values[0, :, 1]
            else:
                sv = shap_values[0]

            # Find top impactful features
            abs_indices = np.argsort(np.abs(sv))[::-1]
            
            rationales = []
            seen_titles = set()
            
            for idx in abs_indices:
                feat_name = self.feature_names[idx]
                impact = float(sv[idx])
                direction = "supports_ai" if impact > 0 else "supports_human"
                
                # Check for friendly mapped explanations
                matched_key = None
                for key in FEATURE_EXPLANATIONS:
                    if key in feat_name:
                        matched_key = key
                        break
                        
                if matched_key:
                    title = FEATURE_EXPLANATIONS[matched_key]["title"]
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)
                    desc = (
                        FEATURE_EXPLANATIONS[matched_key]["ai_desc"]
                        if direction == "supports_ai"
                        else FEATURE_EXPLANATIONS[matched_key]["human_desc"]
                    )
                elif "lfcc" in feat_name:
                    title = "Linear Cepstral High-Frequency Artifacts"
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)
                    desc = (
                        "High-frequency phase and spectral discontinuities detected in linear filterbanks."
                        if direction == "supports_ai"
                        else "Linear filterbanks show natural biological frequency decay."
                    )
                elif "mfcc" in feat_name:
                    title = "Formant Spectral Envelope"
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)
                    desc = (
                        "Formant transitions exhibit synthetic neural interpolation patterns."
                        if direction == "supports_ai"
                        else "Formant bandwidth matches natural human vocal tract resonance."
                    )
                else:
                    title = feat_name.replace("_", " ").title()
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)
                    desc = (
                        f"Acoustic deviation ({feat_name}) aligns with synthetic audio models."
                        if direction == "supports_ai"
                        else f"Acoustic metric ({feat_name}) aligns with human speech baselines."
                    )
                    
                rationales.append({
                    "feature": title,
                    "description": desc,
                    "direction": direction,
                    "impact_score": round(abs(impact), 4)
                })
                
                if len(rationales) >= top_k:
                    break
                    
            return rationales
            
        except Exception as e:
            logger.warning(f"SHAP explanation computation failed: {e}")
            return [
                {
                    "feature": "Acoustic Consistency Analysis",
                    "description": "Multi-feature cepstral and prosodic model evaluation.",
                    "direction": "neutral",
                    "impact_score": 0.5
                }
            ]

    def analyze_chunks(
        self,
        chunks: List[np.ndarray],
        speaker_id: str = "speaker_0",
        voiced_duration_sec: float = 0.0
    ) -> Dict[str, Any]:
        """
        Runs chunk-level inference, score aggregation, SHAP rationale generation,
        and deterministic defensive verdict assignment.
        """
        if not chunks:
            return {
                "speaker_id": speaker_id,
                "risk_score": 0.5,
                "verdict": settings.VERDICT_CANT_TELL,
                "confidence": 0.0,
                "chunk_count": 0,
                "chunk_scores": [],
                "rationales": [{
                    "feature": "Speech Detection",
                    "description": "No sufficient voiced audio detected to perform evaluation.",
                    "direction": "neutral",
                    "impact_score": 0.0
                }]
            }

        feature_matrix = []
        chunk_scores_list = []
        
        for i, chunk in enumerate(chunks):
            feat_vec, _, _ = extract_chunk_features(chunk, sr=settings.SAMPLE_RATE)
            feature_matrix.append(feat_vec)
            
            # Predict probability for AI class (index 1)
            prob_fake = float(self.model.predict_proba(feat_vec.reshape(1, -1))[0][1])
            start_t = i * settings.CHUNK_DURATION_SEC
            end_t = start_t + settings.CHUNK_DURATION_SEC
            
            chunk_scores_list.append({
                "chunk_index": i + 1,
                "start_time_sec": round(start_t, 2),
                "end_time_sec": round(end_t, 2),
                "score": round(prob_fake, 4)
            })

        feature_matrix_np = np.array(feature_matrix)
        raw_scores = [c["score"] for c in chunk_scores_list]
        
        # Aggregation:
        # Deepfake audio attacks frequently splice cloned segments into natural conversation.
        # Therefore we combine mean probability with peak anomaly (90th percentile).
        mean_score = float(np.mean(raw_scores))
        peak_score = float(np.percentile(raw_scores, 90) if len(raw_scores) > 1 else raw_scores[0])
        aggregated_risk = 0.65 * mean_score + 0.35 * peak_score
        aggregated_risk = float(np.clip(aggregated_risk, 0.0, 1.0))
        
        # Model certainty/confidence measure [0.0 - 1.0]
        # High confidence when risk is clearly near 0 or 1, lower when near boundary 0.5
        confidence = float(min(1.0, abs(aggregated_risk - 0.5) * 2.0 + 0.2))

        # Deterministic verdict mapping
        verdict = self.compute_verdict(aggregated_risk, len(chunks), voiced_duration_sec)
        
        # SHAP rationales
        rationales = self.generate_shap_rationales(feature_matrix_np, top_k=3)
        
        return {
            "speaker_id": speaker_id,
            "risk_score": round(aggregated_risk, 4),
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "chunk_count": len(chunks),
            "chunk_scores": chunk_scores_list,
            "rationales": rationales
        }
