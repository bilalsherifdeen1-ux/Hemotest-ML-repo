"""HemoTest FastAPI REST API."""
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from pathlib import Path
from api.schemas import PredictFeaturesRequest, PredictResponse, HealthResponse
from hemotest import HemoTestPredictor

app = FastAPI(
    title="HemoTest ML API — EQUIDX AI Biotech",
    description="Smartphone Hb prediction using WHO 2024 anaemia classification. "
                "NeuroVitalis Health Innovations — neurovitalis.ng",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = Path("models/ensemble_v1.pkl")
predictor: HemoTestPredictor | None = None


@app.on_event("startup")
async def startup():
    global predictor
    if MODEL_PATH.exists():
        predictor = HemoTestPredictor(str(MODEL_PATH))
    else:
        import warnings
        warnings.warn(f"Model not found at {MODEL_PATH}. Run: python scripts/train.py --generate-data")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(status="ok", model_loaded=predictor is not None,
                          version="1.0.0", who_guideline="WHO 2024 Haemoglobin Cutoffs")


@app.post("/predict/image", response_model=PredictResponse, tags=["Prediction"])
async def predict_image(
    file: UploadFile = File(...),
    sex: str = Query("female"), pregnant: bool = Query(False),
    trimester: int | None = Query(None, ge=1, le=3),
    altitude_m: float = Query(0.0), smoker: bool = Query(False),
):
    if predictor is None:
        raise HTTPException(503, "Model not loaded. Run scripts/train.py --generate-data")
    if file.content_type not in ("image/jpeg","image/png","image/jpg"):
        raise HTTPException(400, f"Unsupported: {file.content_type}")
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image.")
    return PredictResponse(**predictor.predict(
        img, sex=sex, pregnant=pregnant, trimester=trimester,
        altitude_m=altitude_m, smoker=smoker,
    ))


@app.post("/predict/features", response_model=PredictResponse, tags=["Prediction"])
async def predict_features(req: PredictFeaturesRequest):
    if predictor is None:
        raise HTTPException(503, "Model not loaded.")
    X = np.array(req.features, dtype=np.float32).reshape(1, -1)
    hb = float(np.clip(predictor.model.predict(X)[0], 2.0, 22.0))
    result = predictor.classifier.classify(hb, sex=req.sex,
                                           pregnant=req.pregnant, trimester=req.trimester)
    return PredictResponse(**result.to_dict())
