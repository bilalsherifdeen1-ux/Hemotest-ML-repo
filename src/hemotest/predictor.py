"""HemoTestPredictor — Main prediction interface (image -> Hb -> WHO classification)."""
from __future__ import annotations
import numpy as np
import cv2
import joblib
from pathlib import Path
from typing import Optional
from loguru import logger
from hemotest.features.colorimetry import ColorimetryExtractor
from hemotest.clinical.classification import AnemiaClassifier, AnemiaResult


class HemoTestPredictor:
    """
    End-to-end predictor: test strip image -> haemoglobin -> anaemia classification.

    Usage:
        p = HemoTestPredictor("models/ensemble_v1.pkl")
        img = cv2.imread("strip.jpg")
        result = p.predict(img, sex="female", pregnant=True, trimester=2)
    """
    def __init__(self, model_path: str | Path, g_reference: float = 200.0):
        self.model = joblib.load(model_path)
        self.extractor = ColorimetryExtractor(g_reference=g_reference)
        self.classifier = AnemiaClassifier()
        logger.info(f"HemoTestPredictor loaded from {model_path}")

    def predict(
        self, image: np.ndarray,
        sex: str = "female", pregnant: bool = False,
        trimester: Optional[int] = None, age_months: Optional[int] = None,
        altitude_m: float = 0.0, smoker: bool = False,
    ) -> dict:
        """Predict Hb and classify anaemia from test strip BGR image."""
        features = self.extractor.extract(image)
        hb = float(np.clip(self.model.predict(features.to_array().reshape(1,-1))[0], 2.0, 22.0))
        result: AnemiaResult = self.classifier.classify(
            hb, sex=sex, pregnant=pregnant, trimester=trimester,
            age_months=age_months, altitude_m=altitude_m, smoker=smoker,
        )
        out = result.to_dict()
        out["diagnostic_features"] = {
            "optical_density": round(features.optical_density, 4),
            "g_mean": round(features.g_mean, 1),
            "l_star": round(features.l_star, 2),
        }
        return out
