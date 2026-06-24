import sys
import time
import io
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.ensemble import ResNetLightGBMEnsemble
except ImportError:
    from ensemble import ResNetLightGBMEnsemble

app = FastAPI(
    title="Audio Deepfake & Voice Spoof Detection API",
    description="ASVspoof logical access deepfake detection REST API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESNET_WEIGHTS = PROJECT_ROOT / "models" / "resnet_spoof_detector.pth"
LGB_WEIGHTS = PROJECT_ROOT / "models" / "lightgbm_ensemble.pkl"

ensemble = ResNetLightGBMEnsemble()


@app.on_event("startup")
def load_models_on_startup():
    ensemble.load_pipeline(
        resnet_path=str(RESNET_WEIGHTS),
        lgb_path=str(LGB_WEIGHTS)
    )
    print("[+] Model pipeline loaded successfully.")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=(ensemble.resnet is not None),
        device=ensemble.device,
        version="1.0.0"
    )


@app.post("/predict", tags=["Prediction"])
async def predict_audio(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        start_t = time.time()
        contents = await file.read()

        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        result = ensemble.predict_audio(contents)
        latency_ms = round((time.time() - start_t) * 1000, 2)

        result.pop("mel_spectrogram", None)
        result["filename"] = file.filename
        result["processing_time_ms"] = latency_ms

        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio processing error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
